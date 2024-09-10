# ESCAPE: Epidemic Simulator Compiler and Programming Environment


The *Epidemic Simulator Compiler and Programming Environment* or ESCAPE
is a framework for efficiently developing
high-performance agent based epidemic simulators.

ESCAPE uses a domain specific language
called the *Epidemic Simulator Language* or ESL,
which users have to use to define the epidemic simulations.
The ESL language provides domain specific constructs
to define compartmental disease models
and the structure of the contact networks
on top of which the disease propagate.
Additionally, ESL also includes general purpose programming constructs
--- conditionals, loops, functions, variables, etc. ---
and parallel constructs
--- select, sample, apply, and reduce ---
for describing interventions.

ESCAPE provides a compiler that converts
the epidemic simulators written in ESL into C++ or CUDA programs.
Simulators created with ESCAPE are high performance
parallel programs that run on multi-core CPU systems
or GPU based systems.
The ESCAPE compiler itself is written in Python.
