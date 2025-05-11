# ResPlan implementation with UP

This is the implementation of the ResPlan algorithm described in this [paper](https://icaps24.icaps-conference.org/program/workshops/keps-papers/KEPS-24_paper_6.pdf).
The implementation is done using the [Unified Planning](https://github.com/aiplan4eu/unified-planning) library for defining the problem and the domain to solve and the [UP-fast-downward](https://github.com/aiplan4eu/up-fast-downward) as the planner.

The algorithm is defined in the `resplan.py` file.

The algorithm was tested on the problem defined in the paper, the test is available in the file `test.py`.
