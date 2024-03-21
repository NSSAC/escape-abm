; Type
(enum name: (identifier) @type)
(global type: (reference (identifier)) @type)

(node_field type: (reference (identifier)) @type)
(edge_field type: (reference (identifier)) @type)

(contagion_state_type type: (reference (identifier) @type))

(function rtype: (reference (identifier)) @type)
(function_param type: (reference (identifier)) @type)

(inline_expression_function type: (reference (identifier)) @type)
(inline_expression_function rtype: (reference (identifier)) @type)
(inline_update_function type: (reference (identifier)) @type)

(variable type: (reference (identifier)) @type)

; Constant

(enum const: (identifier) @constant
      ("," const: (identifier) @constant)*
)
(transition
  entry: (reference (identifier)) @constant
  exit: (reference (identifier)) @constant
)
(transmission
  contact: (reference (identifier)) @constant
  entry: (reference (identifier)) @constant
  exit: (reference (identifier)) @constant
)

; Annotation
(node_annotation) @annotation
(edge_annotation) @annotation

; Function
(normal_dist name: (identifier) @function)
(discrete_dist name: (identifier) @function)
(uniform_dist name: (identifier) @function)

(function name: (identifier) @function)
(function_call function: (reference (identifier) @function.builtin .))

(contagion_function function: (reference (identifier)) @function)
(transition p: (reference (identifier)) @function)
(transition dwell: (reference (identifier)) @function)

; Builtin Reference

((identifier) @constant.builtin
  (#match? @constant.builtin "^(NUM_TICKS|CUR_TICK|NUM_NODES|NUM_EDGES|ALL_NODES|ALL_EDGES)$"))

((identifier) @function.builtin
  (#match? @function.builtin "^(len)$"))

; Literals
[
  (boolean)
] @boolean

[
  (integer)
  (float)
] @number

(template_block) @preproc
(template_variable) @preproc

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
 "<-"
] @operator

[
 "ABSOLUTE" "RELATIVE"
] @constant.builtin


[
 "(" ")"
 "[" "]"
 "{" "}"
] @punctuation.bracket

[
 "->"
 "=>"
 ":"
 ","
 ";"
] @punctuation.special

[
 "enum" "global" "config"
 "node" "edge"
 "distribution" "discrete" "normal" "uniform"
 "p" "v" "mean" "std" "min" "max" "low" "high"
 "contagion"
 "susceptibility" "infectivity" "transmissibility" "enabled"
 "transition" "transmission" "dwell"
 "nodeset" "edgeset"
 "end"
 "def"
 "pass"
 "return"
 "if" "elif" "else"
 "switch" "case" "default"
 "while"
 "var"
 "print"
 "select" "sample" "apply" "reduce"
] @keyword
