; Config
(config name: (identifier) @constant.builtin)

; Enum
(enum name: (identifier) @type)
(enum
    const: (identifier) @constant.builtin
    ("," const: (identifier) @constant.builtin)*
)

; Distributions

(discrete_dist "discrete" @keyword)
(discrete_dist name: (identifier) @function)
(discrete_pv ["p" "v"] @keyword)
(discrete_pv (identifier) @constant.builtin)

(normal_dist ["normal" "mean" "std"] @keyword)
(normal_dist name: (identifier) @function)
(normal_dist mean: (identifier) @constant.builtin)
(normal_dist std: (identifier) @constant.builtin)

(uniform_dist ["uniform" "low" "high"] @keyword)
(uniform_dist name: (identifier) @function)
(uniform_dist low: (identifier) @constant.builtin)
(uniform_dist high: (identifier) @constant.builtin)

; Contagion

(contagion_state_type "state type" @keyword)
(contagion_dwell_type "dwell type" @keyword)

(transition entry: (identifier) @constant.builtin)
(transition exit: (identifier) @constant.builtin)
(transition p: (identifier) @constant.builtin)
(transition dwell: (identifier) @function)
(transition ["p" "dwell"] @keyword)

(transmission (identifier) @constant.builtin)

; Function

(function name: (identifier) @function)

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

(print_statement "print" @keyword)

; General

(type (identifier) @type)
(node_annotation) @annotation
(edge_annotation) @annotation

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
 "config"
 "enum"
 "node"
 "edge"
 "distribution"
 "contagion"
 "transition"
 "transmission"
 "def"
 "pass"
 "return"
 "if" "elif" "else"
 "for" "in"
 "switch" "case" "default"
 "nodeset" "edgeset" "foreach"
] @keyword
