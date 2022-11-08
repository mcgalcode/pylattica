# pylattica - A Framework for Lattice and Cellular Automata Simulation

![test workflow](https://github.com/github/docs/actions/workflows/testing.yaml/badge.svg)


pylattica is a Python library for prototyping and constructing cellular automaton and lattice models. The core features of these models are:

- There is a simulation state that evolves over time by repeatedly applying some unchanging rule
- The state of the simulation has a topology defined by a network of sites (i.e. each site has an unchanging set of neighbor sites)
- Each site has a state value associated with it that could change at each simulation step
- The future state of a site is determined by the state of its neighbor sites

These rules capture many common models in chemistry and materials science. For instance, in the Ising Model, spins are updated with probabilities related to the neighboring spins. In a Lattice Gas Automaton, the velocities of particles are determined collisions with neighboring particles. In lattice Monte Carlo simulations of surface catalysis, adsorption, desorption, and surface diffusion are dependent on the occupancy of neighboring sites.

pylattica aims to provide a general framework for prototyping these types of lattice simulations. It prioritizes providing a straightforward method for experimenting with different interaction rules and interaction neighborhoods. It provides some simple utilities for analyzing simulation states, and in the case of square grid systems, it provides visualization tools for the system state itself. Additionally, since this tool is focused on materials science, there is functionality for mapping system states to CIF files (for use in crystal lattice simulations).

## Debugging

### grpcio

On M1 Macs, importing code related to the reaction-network can cause breakages. The suggested action from the error message, reproduced below, should fix any issues.

```
Failed to import grpc on Apple Silicon. On Apple Silicon machines, try `pip uninstall grpcio; conda install grpcio`. Check out https://docs.ray.io/en/master/ray-overview/installation.html#m1-mac-apple-silicon-support for more details.
```
