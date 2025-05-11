from resplan_last import *
from unified_planning.io import PDDLReader, PDDLWriter
import time

def define_problem_paper():
    # DOMAIN
    Location = UserType("Location") # define the obj type for the problem

    # predicates
    at = Fluent("at", BOOL, obj=Location) # active location for the agent
    car_connected = Fluent("car_connected", BOOL, start=Location, finish=Location) # two location connected by a car
    train_connected = Fluent("train_connected", BOOL, start=Location, finish=Location) # two location connected by a train
    plane_connected = Fluent("plane_connected", BOOL, start=Location, finish=Location) # two location connected by a plane

    # actions
    ## car actions
    move_car = InstantaneousAction("move_car", start=Location, finish=Location)
    move_car.add_precondition(at(move_car.start))
    move_car.add_precondition(car_connected(move_car.start, move_car.finish))
    move_car.add_effect(at(move_car.finish), True)
    move_car.add_effect(at(move_car.start), False)

    # ## train actions
    move_train = InstantaneousAction("move_train", start=Location, finish=Location)
    move_train.add_precondition(at(move_train.start))
    move_train.add_precondition(train_connected(move_train.start, move_train.finish))
    move_train.add_effect(at(move_train.finish), True)
    move_train.add_effect(at(move_train.start), False)

    # ## plane actions
    move_plane = InstantaneousAction("move_plane", start=Location, finish=Location)
    move_plane.add_precondition(at(move_plane.start))
    move_plane.add_precondition(plane_connected(move_plane.start, move_plane.finish))
    move_plane.add_effect(at(move_plane.finish), True)
    move_plane.add_effect(at(move_plane.start), False)

    # PROBLEM
    problem = Problem("navigation problem")

    # add objects
    for i in "ABCDEFG":
        problem.add_object(i, Location)

    # add fluents
    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(car_connected, default_initial_value=False)
    problem.add_fluent(train_connected, default_initial_value=False)
    problem.add_fluent(plane_connected, default_initial_value=False)

    # add actions
    problem.add_action(move_car)
    problem.add_action(move_train)
    problem.add_action(move_plane)

    # add initial state
    problem.set_initial_value(at(problem.object("A")), True)

    # connections
    car_connections = [("A", "B"), ("B", "D"), ("D", "G"), ("B", "C"), ("C", "D"), ("C", "E"), ("E", "D"), ("D", "F")]
    train_connections = [("A", "F"), ("B", "F"), ("D", "G"), ("F", "G")]
    plane_connections = [("A", "C"), ("E", "G")]

    # car connected
    for start, finish in car_connections:
        problem.set_initial_value(car_connected(problem.object(start), problem.object(finish)), True)

    # train connected
    for start, finish in train_connections:
        problem.set_initial_value(train_connected(problem.object(start), problem.object(finish)), True)

    # plane connected
    for start, finish in plane_connections:
        problem.set_initial_value(plane_connected(problem.object(start), problem.object(finish)), True)

    # add goal
    problem.add_goal(at(problem.object("G")))
    
    return problem

def define_problem_from_pddl():
    BLOCK = 'blocksworldMA'
    LOG = 'logistics'
    ROC = 'rocket'

    dir = 'domains'
    domain_name = 'domain.pddl'
    
    # TO CHANGE
    problem_name = BLOCK
    problem_file = 'pfilep6.pddl'

    reader = PDDLReader()
    domain_path = f'resplan/{dir}/{problem_name}/{domain_name}'
    problem_path =f'resplan/{dir}/{problem_name}/{problem_file}'
    return reader.parse_problem(domain_path, problem_path)

def run():
    pddl_problem = define_problem_paper()
    
    K=2
    resilient_problem = ResilientProblem(pddl_problem, K)
    resplan = ResPlan(resilient_problem)
    
    start_time = time.time()
    up.shortcuts.get_environment().credits_stream = None
    solution = resplan.solve()
    
    print("Time: ", time.time() - start_time)
    if solution[0] == "solved":
        print("Plan: ", solution[1])
        print("Policy: ", solution[2])
    else:
        print("No solution found")

if __name__ == '__main__':
    run()
    


