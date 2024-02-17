TODOs
=====

Base features
..............

* Suport parallel node/edge reductions
* Add builtin function to get nodeset/edgeset size
* Make non-discrete distributions parameterizeable

Performance improvement
.......................

* Use bitset/succinct datastructure instead of array of bytes
* Profile cpu code with vtune/advisor

* Merge back to back parallel calls
* SIMD'ize the thread specific loops and node/edge functions

Pipeline
........

* Add pandas read support for input.h5 and output.h5
* Create Parsl based pipeline for running studies


* Add distributed memory (MPI) support
* Change parser from tree-sitter to lark
* Rewrite codegen 

