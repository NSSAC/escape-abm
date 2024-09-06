TODOs
=====

Bugfix
......

* Upgrade to latest tree-sitter parser
* Allow negative constants for config and globals
* Add support to save specific global/config values each tick
* Add more support for templated identifier and expressions

Base features
..............

* Support grouped reductions
* Support codegen for Kokkos / CUDA
* Support distributed memory implementation using MPI / UPCXX

Performance improvement
.......................

* Use bitset/succinct datastructure instead of array of bytes
* Profile cpu code with vtune/advisor

* Merge back to back parallel calls
* Use differnt / faster random number generator

