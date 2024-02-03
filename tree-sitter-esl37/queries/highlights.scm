; Enum
(enum
    "enum" @keyword
    name: (identifier) @type
)
(enum
    const: (identifier) @constant
    ("," const: (identifier) @constant)*
)

; Config
(config
    "config" @keyword
    name: (identifier) @constant
    type: (identifier) @type
)

; Global
(global
    "global" @keyword
    name: (identifier) @constant
    type: (identifier) @type
)


; Node
(node "node" @keyword)
(node_field
  type: (identifier) @type
)
(node_annotation) @annotation

; Edge

(edge "edge" @keyword)
(edge_field
  type: (identifier) @type
)
(edge_annotation) @annotation

; Distributions

[ "distribution" ] @keyword

(constant_dist
    ["constant" "v"] @keyword
    name: (identifier) @function
)

(discrete_dist
    "discrete" @keyword
    name: (identifier) @function
)
(discrete_pv
    ["p" "v"] @keyword
)

(normal_dist
    ["normal" "mean" "std" "min" "max"] @keyword
    name: (identifier) @function
)

(uniform_dist
    ["uniform" "low" "high"] @keyword
    name: (identifier) @function
)

; Contagion

[ "contagion" "transition" "transmission" ] @keyword

(contagion_state_type "state type" @keyword)
(contagion_function "fn" @keyword)
(contagion_function ["susceptibility" "infectivity" "transmissibility" "enabled"] @keyword)
(contagion_function function: (identifier) @function)

(transition
    entry: (identifier) @constant
    exit: (identifier) @constant
    dwell: (identifier) @function
)
(transition ["p" "dwell"] @keyword)

(transmission
  (identifier) @constant
)

; Nodeset / Edgeset
[ "nodeset" "edgeset" ] @keyword

; Function
(function name: (identifier) @function)
(function return: (identifier) @type)
(function_param type: (identifier) @type)

; Function call
(function_call
  function: (reference (identifier) @function .)
)

(
 (function_call
    function: (reference (identifier) @function.builtin .)
 )
 (#match? @function.builtin "^len$")
)

; Print statement

(print_statement "print" @function.builtin)

; select_*
[ "select" "using" "sample" "approx" "relative" "from" ] @keyword

(select_using
  function: (identifier) @function
)

; foreach
[ "foreach" "in" "run" ] @keyword

(foreach_statement
  function: (identifier) @function
)

; Literals
[
  (boolean)
] @boolean

[
  (integer)
  (float)
] @number

(comment) @comment
(string) @string

[
 "+"
 "-"
 "*"
 "/"
 "%"
 "or"
 "and"
 "not"
 "=="
 "!="
 ">"
 ">="
 "<="
 "<"
 "="
 "*="
 "/="
 "%="
 "+="
 "-="
] @operator

[
 "->"
 "=>"
 ":"
 ","
 ";"
 "("
 ")"
] @punctuation.special


[
 "end"
 "def"
 "pass"
 "return"
 "if" "elif" "else"
 "while"
 "var"
] @keyword
