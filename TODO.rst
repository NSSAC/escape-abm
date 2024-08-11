TODOs
=====

Bugfix
......

* Upgrade to latest tree-sitter parser
* Allow negative constants for config and globals
* Use dirty bit instead of preivous state
  for managing intervention related state changes
* Write config values to output file
* Add support to save specific global/config values each tick

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

