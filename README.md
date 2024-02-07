# pylattica - A Framework for Lattice and Cellular Automata Simulation

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mcgalcode/pylattica/testing.yaml?branch=master)
[![Codecov](https://img.shields.io/codecov/c/github/mcgalcode/pylattica?style=for-the-badge)](https://app.codecov.io/gh/mcgalcode/pylattica)

pylattica is a Python library for prototyping and constructing cellular automaton and lattice models. The core features of these models are:

- There is a simulation state that evolves over time by repeatedly applying some unchanging rule
- The state of the simulation has a topology defined by a network of sites (i.e. each site has an unchanging set of neighbor sites)
- Each site has a state value associated with it that could change at each simulation step
- The future state of a site is determined by the state of its neighbor sites

These rules capture many common models in chemistry and materials science. For instance, in the Ising Model, spins are updated with probabilities related to the neighboring spins. In a Lattice Gas Automaton, the velocities of particles are determined collisions with neighboring particles. In lattice Monte Carlo simulations of surface catalysis, adsorption, desorption, and surface diffusion are dependent on the occupancy of neighboring sites.

pylattica aims to provide a general framework for prototyping these types of lattice simulations. It prioritizes providing a straightforward method for experimenting with different interaction rules and interaction neighborhoods. It provides some simple utilities for analyzing simulation states, and in the case of square grid systems, it provides visualization tools for the system state itself. Additionally, since this tool is focused on materials science, there is functionality for mapping system states to CIF files (for use in crystal lattice simulations).

## Installation

`pylattica` can be installed from the [PyPi source](https://pypi.org/project/pylattica/) by running:

```
pip install pylattica
```

It can also be installed by cloning this repository, then running the following in the root of the repository:

```
pip install .
```

### Documentation

Detailed documentation for this library can be found [here](https://mcgalcode.github.io/pylattica/).

### Jupyter Notebook Examples

Example notebooks are included in [docs/guides](https://github.com/mcgalcode/pylattica/tree/master/docs/guides).

### Note about Windows

`pylattica` makes use of Python's fork functionality in the multiprocessing library. This functionality is not available on windows, so certain features (the `parallel` keyword for the `SynchronousRunner`) will not be available on Windows platforms.

## Development

### Installation for development

To install pylattica for development purposes, use the standard editable installation command:

```
pip install -e .
```

from the root of the repository.

### Running tests

After you have installed the repository, ensure that `pytest` is installed, then run:

```
pytest
```

from the root of the repository.


### Building Documentation

The docs for this project are built using mkdocs. To build the documentation

```
pip install '.[docs]'
mkdocs build
```

To run the documentation server locally:

```
mkdocs serve
```

### Linting

This project uses the `black` package for style and formatting, and `prospector` for type checking and other lint warnings. These packages are not listed as dependencies of this project, so you can install them manually. This is partially because this project doesn't rely on specific versions of them, and we expect developers to have their own installations already. You can run them as follows:

To assess the changes that will happen if you run the `black` linter, run the following:

```
black --check src
```

To automatically make the changes, remove the `--check` flag:

```
black src
```

To run all other linters with prospector, use this:

```
prospector
```

In the top of this repository.
