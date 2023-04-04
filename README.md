# ELLib
A c++ software library containing energy landscape methods utilising MPI parallelisation.

Currently implemented:
- Genetic algorithm
- NEB
- BITSS

Minimisation (in the minim sub-library):
- L-BFGS
- FIRE
- Gradient descent
- Simulated annealing

## Installation and use
Download the repository and call `make`.

To use in a program, compile with the `-lellib` flag.
For example: `mpic++ -I$(ELLIB)/include -L$(ELLIB)/bin -lellib -DPARALLEL script.cpp -o run.exe`

Refer to the examples for a simple demonstration of how to use the library.
