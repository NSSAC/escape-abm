# Modeling in ESCAPE

ESCAPE based simulators
simulate multiple concurrent complex contagions
on top of a contact network.
ESL provides domain specific constructs
for easily describing the contact network schema and contagion models.

Agent behavior and public health policies
are implemented in ESCAPE simulators using interventions.
Interventions are procedures that can modify the contact network's
node and edge attributes
as well as parameters of the contagion models.
The ESL language includes general purpose serial programming constructs
and relational algebra inspired parallel programming constructs
to implement these interventions.

ESCAPE simulators are discrete time simulators.
The simulations progress in discrete time steps, called *ticks*.
At each tick, the simulator executes
the programmer defined intervention function
followed by the inter-node transmission
and intra-node transition steps of the contagions.

## Contact network

ESCAPE simulators assume that the contact network $G = (V, E)$,
on top of which the contagions propagate,
is a directed multi graph.
That is, there may be zero or more directed edges
between any given pair of nodes.
Each node and edge may have
an arbitrary number of attributes attached to them.
The attributes may be either `static` or `dynamic`.
Only the dynamic attributes may be modified during the course of a simulation.

From the programmer's perspective,
the nodes and edges, along with their attributes,
are stored in tabular data structures.
Each row in the `node table` represents a node,
with each column representing one of the node's attributes.
The programmer must declare one of the attributes to be the `node key`,
which is assumed to be static.
ESCAPE assumes that this attribute uniquely identifies each node.
Similar to the node table, the edges are stored in the `edge table`.
Each row in this table represents an edge
and each column represents one of the edge's attributes.
The edge table has two required static attributes,
the `source node key` and the `target node key`.
These reference the node table's `node key` attribute
and are used identify the source node $u$ and target node $v$
for each edge $e = (u, v) \in E$.

ESCAPE expects the simulator user to provide
the data for the node and edge tables ---
only the static attributes ---
at runtime.
Dynamic attributes are initialized
to the column's `zero value`
at the beginning of the simulation.[^1]

[^1]: For numeric columns the `zero value` is `0`;
    for boolean columns the zero value is `False`;
    for enumerated types the zero value is the constant
    that is listed first in the enumerated type's definition.

## Contagion

In ESCAPE simulations,
each contagion is represented by a compartmental model
with a finite set of states $S$.
From the programmer's perspective,
each contagion adds a dynamic attribute to the node table
representing the node's state
with respect to that contagion.
Associated with each contagion are
inter-node transmission rules
and intra-node transition rules.

### Inter-node transmission

Each contagion's transmission model in ESCAPE is specified using:
* a set of contact rules: $C = \{ (i, j) \} \quad i,j \in S$,
* a resulting state function: $\kappa: C \to S$,
* a susceptibility function: $\sigma: V \to [0, 1]$,
* an infectivity function: $\iota: V \to [0, 1]$,
* and a transmissibility function: $\tau: E \to [0, 1]$.

Consider an edge $e = (u, v)$
where the source node $u$ is in state $i \in S$,
and the target node $v$ is in state $j \in S$.
This edge is active for transmission if $(i, j) \in C$.
The probability of successful transmission
on an active edge is then given by:
$$P(u, v) = \iota(u) \cdot \sigma(v) \cdot \tau(u, v)$$
Note that, in ESCAPE's contagion model,
the contagion state of source and target nodes of an edge
determine activation.
However, all attributes of the source and target nodes ---
including contagion states ---
are available for modeling the source node's infectivity,
the target node's susceptibility, and the edge's transmissibility.
Additionally, the transmissibility function
also has access to the edge attributes.
On successful transmission the resulting state function $\kappa$
is used to compute the resulting state of the target node.

From the programmer's perspective,
if there are multiple active incoming edges to a node,
the simulator first randomly shuffles all the edges
and iterates over them to sequentially.
It stops after the first successful transmission
or when the list of edges is exhausted.[^2]

[^2]: Note that this sequential view
    is the abstraction provided to the programmer.
    The implementation is free to parallelize this computation
    while ensuring that sequential consistency is maintained.

### Intra-node transition

Each contagion's intra-node transition model ---
also referred to as the within host progression model,
in disease modeling contexts ---
is specified using as a set of transition rules
which is a set of four tuples: $T = \{ (i, j, \pi, \delta) \}$,
where:
* $i \in S$ is the current state of the node,
* $j \in S$ is the resulting state of the node,
* $\pi: V \to [0, 1]$ is the transition probability function,
* and $\delta: V \to \mathbb{R}^+$ is the transition duration function.

Consider a node $v$ that has just entered the contagion state $i$.
The contagion model, during the transition step,
finds all transition rules in $T$ where $i$ is the starting state.
For each of the transition rules that satisfy this condition,
the model computes its transition probability
using the respective transition probability function $\pi$.
Next, one of the transition tuples is selected at random,
weighted by the computed transition probabilities.
Finally, a transition duration is computed
using the selected rule's transition duration function $\delta$
and node's state is scheduled to change
to the rules resulting state $j$
after that simulated duration.

Note that, the transition probability function $\pi$
and the transition duration function $\delta$
is able to use all of the node's attributes ---
including contagion states ---
to model the transition probability and duration.
Additionally, the duration functions are often stochastic.
ESCAPE provides built in functions representing popular
probability distributions to implement these functions.

## Interventions

Simulators written in ESCAPE use intervention
as the primary method to implement
agent behavior, public policies, and managing global state.
ESCAPE's interventions are heavily inspired by
the intervention system of [EpiHiper],
where it has been used to implement complex scenarios
such as: limited supply vaccinations, city-wide lock downs,
social distancing, masking mandates,
and complex agent behavior ---
including masking and vaccine hesitancy.
When creating ESCAPE simulations
the programmer must provide a function named `intervene`.
The simulations calls this function at the beginning of every tick.
Since ESL includes general purpose programming constructs ---
such as variables, functions, conditionals, and loops ---
the intervene function's modeling capacity is Turing complete
and thus can implement any programming construct.
Additionally, ESL includes structured query language or SQL inspired
constructs to query and update the node and edge tables.
This allows the programmer describe computations
that iterate over, filter, sample and update the node and edge tables.
Finally, this also allows the programmer
to describe aggregate computations (or reductions)
over the node and edge tables.
These facilities make ESCAPE's intervention system
a strict superset of what can be achieved via EpiHiper.

[EpiHiper]: https://github.com/NSSAC/EpiHiper

