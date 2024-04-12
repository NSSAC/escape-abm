TODOs
=====

Base features
..............

* Support grouped reductions
* Support codegen for CUDA / Sycl
* Support distributed memory implementation using MPI / UPCXX

Performance improvement
.......................

* Use bitset/succinct datastructure instead of array of bytes
* Profile cpu code with vtune/advisor

* Merge back to back parallel calls
* SIMD'ize the thread specific loops and node/edge functions
* Use differnt random number generator

Pipeline
........

* Create Ray based pipeline for running studies

* Add distributed memory (MPI) support
