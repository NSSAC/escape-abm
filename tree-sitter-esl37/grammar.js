// Tree sitter grammar file for esl

const PREC = {
    LOGICAL_OR: 1,
    LOGICAL_AND: 2,
    EQUAL: 3,
    RELATIONAL: 4,
    ADD: 5,
    MULTIPLY: 6,
    UNARY: 7,
    CALL: 8,
};

module.exports = grammar({
    name: 'esl37',

    word: $ => $.identifier,

    rules: {
        source_file: $ => repeat(choice(
            $.config,
            $.enum,
            $.node,
            $.edge,
            $.distributions,
            $.contagion,
            $.function,
            $.test_statement,
            $.test_expression
        )),

        identifier: _ => /[_a-zA-Z][_a-zA-Z0-9]*/,

        type: $ => $.identifier,

        reference: $ => dotSep1($.identifier),

        config: $ => seq(
            'config',
            field('name', $.identifier),
            '=',
            field('default', choice($.integer, $.float, $.true, $.false)),
            $._terminator
        ),

        enum: $ => seq(
            'enum',
            field('name', $.identifier),
            field('const', commaSep1($.identifier)),
            'end'
        ),

        node: $ => seq(
            'node',
            field('field', repeat($.node_field)),
            'end'
        ),

        node_field: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.type),
            field('annotation', repeat($.node_annotation)),
            $._terminator
        ),

        node_annotation: _ => token(choice(
            alias(/node\s+key/, 'node key'),
            'static'
        )),

        edge: $ => seq(
            'edge',
            field('field', repeat($.edge_field)),
            'end'
        ),

        edge_field: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.type),
            field('annotation', repeat($.edge_annotation)),
            $._terminator
        ),

        edge_annotation: _ => token(choice(
            alias(/target\s+node\s+key/, 'target node key'),
            alias(/source\s+node\s+key/, 'source node key'),
            'static'
        )),

        distributions: $ => seq(
            'distribution',
            repeat($._distribution),
            'end'
        ),

        _distribution: $ => choice(
            $.discrete_dist,
            $.normal_dist,
            $.uniform_dist
        ),

        discrete_dist: $ => seq(
            'discrete',
            field('name', $.identifier),
            field('pv', repeat($.discrete_pv)),
            'end'
        ),

        discrete_pv: $ => seq(
            'p', '=', field('p', choice($._number, $.identifier)), ',',
            'v', '=', field('v', choice($._number, $.identifier)),
            $._terminator
        ),

        normal_dist: $ => seq(
            'normal',
            field('name', $.identifier),
            'mean', '=', field('mean', choice($._number, $.identifier)), ',',
            'std', '=', field('std', choice($._number, $.identifier)),
            $._terminator,
            'end'
        ),

        uniform_dist: $ => seq(
            'uniform',
            field('name', $.identifier),
            'low', '=', field('low', choice($._number, $.identifier)), ',',
            'high', '=', field('high', choice($._number, $.identifier)),
            $._terminator,
            'end'
        ),

        contagion: $ => seq(
            'contagion',
            field('name', $.identifier),
            field('body', repeat(choice(
                $.contagion_state_type,
                $.contagion_dwell_type,
                $.transitions,
                $.transmissions,
                $.function,
            ))),
            'end'
        ),

        contagion_state_type: $ => seq(
            alias(/state\s+type/, 'state type'),
            field('type', $.type),
            $._terminator
        ),

        contagion_dwell_type: $ => seq(
            alias(/dwell\s+type/, 'dwell type'),
            field('type', $.type),
            $._terminator
        ),

        transitions: $ => seq(
            'transition',
            field('body', repeat($.transition)),
            'end'
        ),

        transition: $ => seq(
            field('entry', $.identifier), '->',
            field('exit', $.identifier), ',',
            'p', '=', field('p', choice($._number, $.identifier)), ',',
            'dwell', '=', field('dwell', $.identifier),
            $._terminator
        ),

        transmissions: $ => seq(
            'transmission',
            field('body', repeat($.transmission)),
            'end'
        ),

        transmission: $ => seq(
            field('contact', $.identifier), '=>',
            field('entry', $.identifier), '->',
            field('exit', $.identifier),
            $._terminator
        ),

        function: $ => seq(
            'def',
            field('name', $.identifier),
            field('params', $.function_params),
            field('return', optional($._function_return)),
            ':',
            field('body', $.function_body),
            'end'
        ),

        function_params: $ => seq(
            '(', commaSep($.function_param), ')'
        ),

        function_param: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.type),
        ),

        _function_return: $ => seq(
            '->',
            $.type
        ),

        function_body: $ => repeat1($._statement),

        test_statement: $ => seq(
            '__test', 'statement', ':',
            repeat1($._statement),
            'end'
        ),

        _statement: $ => choice(
            $.pass_statement,
            $.return_statement,
            $.if_statement,
            $.switch_statement,
            $.while_loop,
            $.for_loop,
            $.call_statement,
            $.assignment_statement,
            $.update_statement,
            $.nodeset_statement,
            $.edgeset_statement,
            $.foreach_loop,
            $.print_statement
        ),

        pass_statement: $ => seq(
            'pass',
            $._terminator
        ),

        return_statement: $ => seq(
            'return',
            $._expression,
            $._terminator
        ),

        if_statement: $ => seq(
            'if',
            field('condition', $._expression),
            ':',
            field('body', repeat1($._statement)),
            field('elif', repeat($.elif_section)),
            field('else', optional($.else_section)),
            'end'
        ),

        elif_section: $ => seq(
            'elif',
            field('condition', $._expression),
            ':',
            field('body', repeat1($._statement))
        ),

        else_section: $ => seq(
            'else', ':',
            field('body', repeat1($._statement))
        ),

        switch_statement: $ => seq(
            'switch',
            field('condition', $._expression),
            ':',
            field('case', repeat1($.case_section)),
            field('default', optional($.default_section)),
            'end'
        ),

        case_section: $ => seq(
            'case',
            field('match', $._expression),
            ':',
            field('body', repeat1($._statement))
        ),

        default_section: $ => seq(
            'default', ':',
            field('body', repeat1($._statement))
        ),

        while_loop: $ => seq(
            'while',
            field('condition', $._expression),
            ':',
            field('body', repeat1($._statement)),
            'end'
        ),

        for_loop: $ => seq(
            'for',
            field('var', $.identifier),
            'in',
            field('range', $.range_expression),
            ':',
            field('body', repeat1($._statement)),
            'end'
        ),

        range_expression: $ => seq(
            'range', '(',
            $._expression,
            optional($._expression),
            optional($._expression),
            ')'
        ),

        call_statement: $ => seq(
            $.function_call,
            $._terminator
        ),

        assignment_statement: $ => seq(
            field('left', $.reference),
            '=',
            field('right', $._expression),
            $._terminator,
        ),

        update_statement: $ => seq(
            field('left', $.reference),
            field('operator', choice(
                '*=',
                '/=',
                '%=',
                '+=',
                '-=',
            )),
            field('right', $._expression),
            $._terminator,
        ),

        nodeset_statement: $ => seq(
            'nodeset',
            field('name', $.identifier),
            '=',
            '{',
            field('var', $.identifier),
            ':',
            field('condition', $._expression),
            '}'
        ),

        edgeset_statement: $ => seq(
            'edgeset',
            field('name', $.identifier),
            '=',
            '{',
            field('var', $.identifier),
            ':',
            field('condition', $._expression),
            '}'
        ),

        foreach_loop: $ => seq(
            'foreach',
            field('var', $.identifier),
            'in',
            field('set', $.reference),
            ':',
            field('body', repeat1($._statement)),
            'end'
        ),

        print_statement: $ => seq(
            'print', '(',
            field('arg', commaSep1(choice($.string, $._expression))),
            ')',
            $._terminator
        ),

        test_expression: $ => seq(
            '__test', 'expression', ':',
            field('expression', $._expression),
            'end'
        ),

        _expression: $ => choice(
            $.integer,
            $.float,
            $.true,
            $.false,
            $.unary_expression,
            $.binary_expression,
            $.parenthesized_expression,
            $.reference,
            $.function_call,
        ),

        unary_expression: $ => prec.left(PREC.UNARY, seq(
            field('operator', choice('not', '-', '+')),
            field('argument', $._expression),
        )),

        binary_expression: $ => {
            const table = [
                ['+', PREC.ADD],
                ['-', PREC.ADD],
                ['*', PREC.MULTIPLY],
                ['/', PREC.MULTIPLY],
                ['%', PREC.MULTIPLY],
                ['or', PREC.LOGICAL_OR],
                ['and', PREC.LOGICAL_AND],
                ['==', PREC.EQUAL],
                ['!=', PREC.EQUAL],
                ['>', PREC.RELATIONAL],
                ['>=', PREC.RELATIONAL],
                ['<=', PREC.RELATIONAL],
                ['<', PREC.RELATIONAL],
            ];

            return choice(...table.map(([operator, precedence]) => {
                return prec.left(precedence, seq(
                    field('left', $._expression),
                    field('operator', operator),
                    field('right', $._expression),
                ));
            }));
        },

        parenthesized_expression: $ => seq(
            '(',
            field('expression', $._expression),
            ')'
        ),

        function_call: $ => prec(PREC.CALL, seq(
            field('function', $.reference),
            '(',
            field('arg', commaSep($._expression)),
            ')'
        )),

        comment: _ => token(seq('#', /.*/)),

        _number: $ => choice($.integer, $.float),

        integer: _ => token(choice(
            seq(
                choice('0x', '0X'),
                repeat1(/_?[A-Fa-f0-9]+/),
                optional(/[Ll]/),
            ),
            seq(
                choice('0o', '0O'),
                repeat1(/_?[0-7]+/),
                optional(/[Ll]/),
            ),
            seq(
                choice('0b', '0B'),
                repeat1(/_?[0-1]+/),
                optional(/[Ll]/),
            ),
            seq(
                repeat1(/[0-9]+_?/),
                optional(/[Ll]/), // long numbers
            ),
        )),

        float: _ => {
            const digits = repeat1(/[0-9]+_?/);
            const exponent = seq(/[eE][\+-]?/, digits);

            return token(seq(
                choice(
                    seq(digits, '.', optional(digits), optional(exponent)),
                    seq(optional(digits), '.', digits, optional(exponent)),
                    seq(digits, exponent),
                ),
                optional(/[Ll]/),
            ));
        },

        true: _ => 'True',
        false: _ => 'False',

        string: _ => token(seq(
            '"',
            repeat(choice(
                token.immediate(/[^\\"\n]+/),
                seq(
                    '\\',
                    choice(
                        /[^xuU]/,
                        /\d{2,3}/,
                        /x[0-9a-fA-F]{2,}/,
                        /u[0-9a-fA-F]{4}/,
                        /U[0-9a-fA-F]{8}/,
                    ),
                )
            )),
            '"'
        )),

        _whitespace: _ => /\s/,

        _terminator: _ => choice('\n', ';'),
    },

    extras: $ => [
        $.comment,
        $._whitespace
    ]
});

function commaSep1(rule) {
    return seq(rule, repeat(seq(',', rule)));
}

function commaSep(rule) {
    return optional(commaSep1(rule));
}

function dotSep1(rule) {
    return seq(rule, repeat(seq('.', rule)));
}

// function dotSep(rule) {
//     return optional(dotSep1(rule));
// }