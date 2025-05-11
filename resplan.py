from queue import LifoQueue
from itertools import chain, combinations
from typing import Set, Type as ClassType, List, Tuple
from unified_planning.shortcuts import *
from unified_planning.model.types import BOOL
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import ActionInstance
from unified_planning.model import UPState
import time

class ResilientProblem:
    '''
    Represent the resilient problem, with the problem and level of resiliency.
    '''

    def __init__(self, problem : Problem, K:int = 0):
        self.problem = problem
        self.K = K

# class for representing the tuple to include in the open list, in the self.R_up and self.R_down sets, with some useful methods and utilities
class ResTupla:
    '''
    The ResTupla class is used to represent the tuple that is included in the open list, in the self.R_up and self.R_down sets.
    We used the class because we found some problems:
    - while adding the tupla to a set, a set cannot be contained in a set, so we need to use frozenset
    - while adding the tupla to the different sets the action were slightly different (different preconditions due to the encaplusulate operation), so we need to overwr the __eq__ and __hash__ methods
    '''

    def __init__(self, state:UPState, k, V):
        self.state = state
        self.k = k
        self.V = frozenset(V)

    def __hash__(self):
        name = ""
        for v in self.V:
            name += f"{v.action.name}"
            for p in v.actual_parameters:
                name += f"{p}"+f"{p.type}"

        return hash((self.state, self.k, name))

    def __eq__(self, other):
        if self.state._values == other.state._values and self.k == other.k:
            if len(self.V) == len(other.V):
                for v1, v2 in zip(self.V, other.V):
                    if v1.action.name != v2.action.name or v1.actual_parameters != v2.actual_parameters:
                        return False
                return True

        return False

    def __repr__(self):
        V_set = set(self.V)
        set_state = self.convert_ups_state(self.state)
        if len(V_set) == 0:
            return f"({set_state}, {self.k}, {{}})"
        else:
            return f"({set_state}, {self.k}, {V_set})"

    def values(self):
        '''
        The values function is used to get the values of the tuple, and return them as a tuple.
        '''
        return self.state, self.k, self.V

    # by now for printing results
    def convert_ups_state(self, ups_state):
        state_set = set()
        up_state = ups_state._values
        for si in up_state:
            if up_state[si].bool_constant_value() == True:
                state_set.add(si)
        return state_set

class ResPlanComputer:
    '''
    The ResPlanComputer class is used to represent the resilient plan solver, with the problem and the number of resilient actions.
    This class has to be extended to a specific planner. It means that we need to overwrite:
    - the encapsulate_problem function, because of how we want to encapsulate the problem
    - the compute_plan function, because of the planner used
    '''

    def __init__(self, s, k, problem, V,  R_down):
        self.problem = problem
        self.encapsulate_problem(problem, s, V)
        self.s = s
        self.V = V
        self.S_down = {s.state for s in R_down if s.k == k}

    def compute(self) -> Tuple[List[ActionInstance], List[UPState]]:
        '''
        The compute function is used to compute the plan and the trajectory.
        For not having errors we need to overwrite the compute_plan and the get_trajectory functions.
        '''
        self.plan = self.compute_plan()
        self.trajectory = self.get_trajectory()
        return self.plan, self.trajectory

    def get_trajectory(self) -> List[UPState]:
        '''
        The get_trajectory function is used to obtain the trajectory of the plan execution.
        We need to extract the trajectory from the plan execution.
        '''

        if self.plan is None:
            return None

        with SequentialSimulator(problem=self.problem) as simulator:
            simulator_state = self.s
            states = [simulator_state]
            problem_plan = [ActionInstance(self.problem.action(a.action.name), a.actual_parameters) for a in self.plan]
            for a in problem_plan:
                simulator_state=simulator.apply(simulator_state, a.action, a.actual_parameters)
                states.append(simulator_state)
            return states

    # the function that needs to be overwritten because of how we want to encapsulate the problem
    def encapsulate_problem(self, problem, s, denied_actions) -> Problem:
        '''
        Specify how to encapsulate the problem as we want for resplan, based on the planner characteristics.
        '''
        raise NotImplementedError

    # the function that needs to be overwritten because of the planner used
    def compute_plan(self) -> List[ActionInstance]:
        '''
        Specify how to compute the plan because of the planner used.
        '''
        raise NotImplementedError

class ResPlanComputerFastDownward(ResPlanComputer):
    '''
    The ResPlanComputerFastDownward class is used to represent the resilient plan solver that uses the fast-downward planner.
    It specifies the encapsulation of the problem.
    '''

    def __init__(self, s, k, problem, V, R_down):
        super().__init__(s, k, problem, V, R_down)
        self.planner = "fast-downward"

    def encapsulate_problem(self, problem, s, V) -> Problem:
        '''
        The encapsulate_problem function is used to encapsulate the problem as we want for resplan, using fast-downward as planner.
        The encapsulation means that we add fluents to deny actions and we set the initial state as we want.
        We need to deny some specific action instances, for doing that we add a fluent that denies the action, and we set it to true for the specific instances, with the specific parameters.
        '''

        self.encapsulated_problem = problem.clone()

        # DENY ACTIONS: add fluents to deny actions
        for action in self.encapsulated_problem.actions:
            param_list = [(param.name, param.type) for param in action.parameters]
            denied_action = Fluent(f"{action.name}_denied", BOOL, **{name: typ for name, typ in param_list})
            self.encapsulated_problem.add_fluent(denied_action, default_initial_value=False)
            action.add_precondition(Not(denied_action(*[a for a in denied_action.signature])))
            # this way if the fluent (with the right parameters) is true the action is denied
            # this is a way for dening action instances

        for a in V:
            fluent_vietato = self.encapsulated_problem.fluent(f"{a.action.name}_denied")
            self.encapsulated_problem.set_initial_value(fluent_vietato(*[p for p in a.actual_parameters]), True)

        # SETTING NEW INITIAL STATE: working on dynamic fluents
        dynamic_fluents = self.get_dynamic_fluents(self.encapsulated_problem)
        state_set = self.convert_ups_state(s, self.encapsulated_problem) # a set of true fluents

        for f in self.encapsulated_problem.initial_values:
            if f.fluent() in dynamic_fluents:
                self.encapsulated_problem.set_initial_value(f, False)

        for f in state_set:
            self.encapsulated_problem.set_initial_value(f, True)

        return self.encapsulated_problem

    def compute_plan(self) -> List[ActionInstance]:
        '''
        We specify how to compute the plan because we first need to compile the problem due state invariants are not supported by the planner.
        '''

        # DENY STATES
        for state in self.S_down:
            # this way a state (set of true fluents) is denied
            set_state = self.convert_ups_state(state, self.problem)
            for action in self.encapsulated_problem.actions:
               action.add_precondition(Not(And(*[s for s in set_state])))

            self.encapsulated_problem.add_goal(Not(And(*[s for s in set_state])))

        # SOLVE PLAN
        with OneshotPlanner(name=self.planner) as planner:
            res = planner.solve(self.encapsulated_problem)
            if res.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                return res.plan.actions
            else:
                return None

    def get_dynamic_fluents(self, problem) -> Set[Fluent]:
        return set(problem.fluents) - problem.get_static_fluents()

    def convert_ups_state(self, ups_state, problem) -> Set[Fluent]:
        dynamic_fluents = self.get_dynamic_fluents(problem)
        state_set = set()
        up_state = ups_state._values
        for si in up_state:
            if si.fluent() in dynamic_fluents and up_state[si].bool_constant_value() == True:
                state_set.add(si)
        return state_set


class ResPlan:
    '''
    The ResPlan class is used to represent the resilient plan, with the problem and the number of resilient actions.
    There are some parts that are domain independent, and some that are domain dependent, hence still to be implemented.
    '''

    # the constructor of the class, with the resilient problem
    def __init__(self, resilient_problem: ResilientProblem, computer: ClassType[ResPlanComputer]=ResPlanComputerFastDownward):
        self.problem = resilient_problem.problem
        self.K = resilient_problem.K
        self.computer = computer  # this has to be a class that extends ResPlanComputer

        # Open list and Sets
        self.open_list = LifoQueue()
        self.R_up = set()
        self.R_down = set()

        # Policy and Applicable actions
        self.policy = dict()
        self.applicable_actions = dict()


    def solve(self) -> Tuple[str, List[ActionInstance], dict]:
        '''
        The solve function is used to solve the resilient plan.
        The function returns:
        - the status of the solution, as a string
        - the plan
        - the policy.
        '''
        # get initial state in the form we want
        with SequentialSimulator(problem=self.problem) as simulator:
            s0 = simulator.get_initial_state()

        # add the initial ResTupla to the open list
        initial_tupla = ResTupla(s0, self.K, {})
        self.open_list.put(initial_tupla)

        # main loop
        while not self.open_list.empty():
            
            tupla = self.open_list.get()
            s,k,V = tupla.values()

            # check if the state is already in self.R_up or self.R_down
            if (tupla not in self.R_up) and (tupla not in self.R_down):
                R_check, action_to_policy = self.Rcheck(s, k, V)
                if R_check: # the state is resilient
                    self.R_up.add(ResTupla(s,k,V))
                    self.update_policy(s, k, V, action_to_policy)
                else:
                    plan, trajectory = self.computer(s, k, self.problem, V, self.R_down).compute()
                    if plan is None: # there is no resilience
                        self.update_non_resilient(s, k, V)
                    elif k >= 1:
                        for i in range(len(plan)):
                            self.open_list.put(ResTupla(trajectory[i], k, V))
                            self.open_list.put(ResTupla(trajectory[i], k - 1, V | {plan[i]}))
                            self.update_applicable_actions(trajectory[i], plan[i])
                        self.R_up.add(ResTupla(trajectory[-1], k, V))
                    else: # k = 0, the states in the plane are 0 resilient
                        for t, p in zip(trajectory[:-1], plan):
                            self.R_up.add(ResTupla(t,0,V))
                            self.update_policy(t, 0, V, p)
                        self.R_up.add(ResTupla(trajectory[-1],0,V))

        if initial_tupla in self.R_up:
            return "solved", self.extract_solution(), self.policy
        else:
            return "unsolvable", None, None

    def Rcheck(self, s, k, V):
        '''
        The Rcheck function is used to check if the state is already classified as resilient.
        '''

        # get all the applicable actions
        applicable_actions = self.get_applicable_actions(s, V)
        # for each action check if the result is in self.R_up and the previous state is in self.R_up
        for a in applicable_actions:
            if (ResTupla(self.get_result(a,s), k, V) in self.R_up) and (ResTupla(s, k-1, V | {a}) in self.R_up):
                return True, a

        return False, None

    # the function that updates the non resilient states, also updates with all the subset of V
    def update_non_resilient(self, state, k, denied_actions) -> None:
        '''
        The update_non_resilient function is used to update the non resilient states.
        For updating the non resilient states we need to add all the tuple in the form <s, k', V?> where V'=subset(V), and k'=K-len(V).
        '''

        subset_denied_actions = chain.from_iterable(combinations(denied_actions, r) for r in range(len(denied_actions) + 1))
        for V in subset_denied_actions:
            self.R_down.add(ResTupla(state, self.K-len(V), V))

    # the function that extracts the solution
    def extract_solution(self) -> List[ActionInstance]:
        '''
        The extract_solution function is used to extract the solution from the policy.
        It uses a sequential simulator and goes only in the states that are proved K resilient, hence if the tuple <s, K, set()> is in R_up.
        '''

        plan = []
        with SequentialSimulator(problem=self.problem) as simulator:
            simulator_state = simulator.get_initial_state()
            while not simulator.is_goal(simulator_state):
                applicable_actions = self.get_applicable_actions(simulator_state, set())
                for a in applicable_actions:
                    s = simulator.apply(simulator_state, a.action, a.actual_parameters)
                    if ResTupla(s, self.K, {}) in self.R_up:
                        plan.append(a)
                        simulator_state = s
                        break
        return plan


    def update_policy(self, s, k, V, a) -> None:
        '''
        The update_policy function is used to update the policy.
        The policy is a data structure where {<s, k, V> : action} is a key-value pair.
        Here we create an ActionInstance with the action from the original problem and the actual parameters, and we add it to the policy with the correspondent tuple.
        '''
        action = ActionInstance(
            self.problem.action(a.action.name),
            tuple([self.problem.object(p.object().name) for p in a.actual_parameters])
            )

        self.policy[ResTupla(s, k, V)] = action

        self.update_applicable_actions(s, a)

    def update_applicable_actions(self, s, a) -> None:
        '''
        The update_applicable_actions function is used to update the applicable actions.
        The applicable actions are a data structure where {s : {action}} is a key-value pair.
        Here we create an ActionInstance with the action from the original problem and the actual parameters, and we add it to the applicable actions with the correspondent state.
        '''

        action = ActionInstance(
            self.problem.action(a.action.name),
            tuple([self.problem.object(p.object().name) for p in a.actual_parameters]))

        if s not in self.applicable_actions.keys():
            self.applicable_actions[s] = set()

        for a in self.applicable_actions[s]:
            if a.is_semantically_equivalent(action):
                return

        self.applicable_actions[s].add(action)

    def get_applicable_actions(self, s, V) -> List[ActionInstance]:
        '''
        The get_applicable_actions function is used to get the applicable actions in a certain state.
        The applicable actions are a data structure where {s : {action}} is a key-value pair.
        Here we get the applicable actions for the state s, and we filter them with the set V.
        '''
        applicable_actions = []

        if s not in self.applicable_actions.keys():
            return applicable_actions

        for a in self.applicable_actions[s]:
            if len(V) > 0:
                for v in V:
                    if not a.is_semantically_equivalent(v):
                        applicable_actions.append(a)
            else:
                applicable_actions.append(a)
        return applicable_actions

    def get_result(self, action, s) -> UPState:
        with SequentialSimulator(problem=self.problem) as simulator:
            new_state = simulator.apply(s, action.action, action.actual_parameters)
            return new_state
