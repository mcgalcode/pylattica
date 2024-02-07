# Getting Started with Pylattica

`pylattica` is an extremely flexible library. It doesn't make many prescriptions regarding the use of the pieces that it implements. It's main goal is to provide implementations of common lattice simulation constructs which you can piece together in whatever ways aid your development use case. As a result, a generic "how to" guide is a difficult proposition.

I believe that the best way to learn how to use `pylattica` is by way of example, so I recommend looking at the [Implementing Conway's Game of Life](./game_of_life.ipynb) example.

It's likely that this demonstration does not illustrate everything you need for your own simulation. To see how each of the core pieces of pylattica work, take a look at the following topic-specific guides:

- [Constructing Lattices](./constructing_lattices.ipynb)
- [Periodic Structures](./periodic_structures.ipynb)
- [Building Neighborhoods](./neighborhoods.ipynb)
- [Representing Simulation State](./neighborhoods.ipynb)
- [Controllers and Update Rules](./controllers_and_update_rules.ipynb)
- [Runners](./runners.ipynb)
- [Working With Square Grids](./square_grids.ipynb)

**NOTE: These guides can be found in [docs/guides](https://github.com/mcgalcode/pylattica/tree/master/docs/guides) in the source repository.**

Finally, if all else fails, the [Reference](../reference/core/lattice.md) provides individual method-level detail for the use of this library.