; Config
(config
    "config" @keyword
    name: (identifier) @constant
    type: (identifier) @type
)

; Enum
(enum
    "enum" @keyword
    name: (identifier) @type
)
(enum
    const: (identifier) @constant
    ("," const: (identifier) @constant)*
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

(transition
    entry: (identifier) @constant
    exit: (identifier) @constant
    dwell: (identifier) @function
)
(transition ["p" "dwell"] @keyword)

(transmission
  (identifier) @constant
)

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

; Literals

[
  (true)
  (false)
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
 "{"
 "}"
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
 "switch" "case" "default"
 "while" "for" "in"
 "var" "const"
 "nodeset" "edgeset" "foreach"
] @keyword
