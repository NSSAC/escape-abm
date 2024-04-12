// Tree sitter grammar file for ESL37

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
            $.enum,
            $.global,
            $.node,
            $.edge,
            $.distributions,
            $.contagion,
            $.nodeset,
            $.edgeset,
            $.function,
            $.test_statement,
            $.test_expression
        )),

        reference: $ => dotSep1($.identifier),

        enum: $ => seq(
            'enum',
            field('name', $.identifier),
            field('const', commaSep1($.identifier)),
            'end'
        ),

        global: $ => seq(
            field('category', choice('global', 'config')),
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            '=',
            field('default', choice($.boolean, $.integer, $.float)),
            $._terminator,
        ),

        node: $ => seq(
            'node',
            field('field', repeat1($.node_field)),
            'end'
        ),

        node_field: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            field('annotation', repeat($.node_annotation)),
            $._terminator
        ),

        node_annotation: _ => token(choice(
            alias(/node\s+key/, 'node key'),
            'static',
            'save'
        )),

        edge: $ => seq(
            'edge',
            field('field', repeat1($.edge_field)),
            'end'
        ),

        edge_field: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            field('annotation', repeat($.edge_annotation)),
            $._terminator
        ),

        edge_annotation: _ => token(choice(
            alias(/target\s+node\s+key/, 'target node key'),
            alias(/source\s+node\s+key/, 'source node key'),
            'static',
            'save'
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
            'p', '=', field('p', $._number), ',',
            'v', '=', field('v', $._expression),
            $._terminator
        ),

        normal_dist: $ => seq(
            'normal',
            field('name', $.identifier),
            'mean', '=', field('mean', $._expression), ',',
            'std', '=', field('std', $._expression),
            optional(seq(',', 'min', '=', field('min', $._expression))),
            optional(seq(',', 'max', '=', field('max', $._expression))),
            $._terminator,
            'end'
        ),

        uniform_dist: $ => seq(
            'uniform',
            field('name', $.identifier),
            'low', '=', field('low', $._expression), ',',
            'high', '=', field('high', $._expression),
            $._terminator,
            'end'
        ),

        contagion: $ => seq(
            'contagion',
            field('name', $.identifier),
            field('body', repeat(choice(
                $.contagion_state_type,
                $.contagion_function,
                $.transitions,
                $.transmissions,
            ))),
            'end'
        ),

        contagion_state_type: $ => seq(
            alias(/state\s+type/, 'state type'),
            field('type', $.reference),
            $._terminator
        ),

        contagion_function: $ => seq(
            field('type', choice('susceptibility', 'infectivity', 'transmissibility', 'enabled')),
            field('function', choice($._expression, $.inline_expression_function)),
            $._terminator
        ),

        transitions: $ => seq(
            'transition',
            field('body', repeat($.transition)),
            'end'
        ),

        transition: $ => seq(
            field('entry', $.reference), '->',
            field('exit', $.reference), ',',
            optional(seq('p', '=', field('p', choice($._expression, $.inline_expression_function)), ',')),
            'dwell', '=', field('dwell', choice($._expression, $.inline_expression_function)),
            $._terminator
        ),

        transmissions: $ => seq(
            'transmission',
            field('body', repeat($.transmission)),
            'end'
        ),

        transmission: $ => seq(
            field('contact', $.reference), '=>',
            field('entry', $.reference), '->',
            field('exit', $.reference),
            $._terminator
        ),

        nodeset: $ => seq(
            'nodeset',
            field('name', commaSep1($.identifier)),
        ),

        edgeset: $ => seq(
            'edgeset',
            field('name', commaSep1($.identifier)),
        ),

        function: $ => seq(
            'def',
            field('name', $.identifier),
            field('params', $.function_params),
            optional(seq('->', field('rtype', $.reference))),
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
            field('type', $.reference),
        ),

        function_body: $ => repeat1($._statement),

        inline_expression_function: $ => seq(
            '[',
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            '->',
            field('rtype', $.reference),
            ']',
            '(',
            field('expression', $._expression),
            ')'
        ),

        inline_update_function: $ => seq(
            '[',
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            ']',
            '{',
            field('stmt', repeat1($.update_statement)),
            '}'
        ),

        test_statement: $ => seq(
            '__test', 'statement', ':',
            repeat1($._statement),
            'end'
        ),

        _statement: $ => choice(
            $.variable,
            $.pass_statement,
            $.return_statement,
            $.if_statement,
            $.switch_statement,
            $.while_loop,
            // $.for_loop,
            $.call_statement,
            $.update_statement,
            $.print_statement,
            $.select_statement,
            $.sample_statement,
            $.apply_statement,
            $.reduce_statement
        ),

        variable: $ => seq(
            'var',
            field('name', $.identifier),
            ':',
            field('type', $.reference),
            '=',
            field('init', $._expression),
            $._terminator,
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

        // for_loop: $ => seq(
        //     'for',
        //     field('var', $.identifier),
        //     'in',
        //     field('range', $.range_expression),
        //     ':',
        //     field('body', repeat1($._statement)),
        //     'end'
        // ),

        // range_expression: $ => seq(
        //     'range', '(',
        //     $._expression,
        //     optional($._expression),
        //     optional($._expression),
        //     ')'
        // ),

        call_statement: $ => seq(
            $.function_call,
            $._terminator
        ),

        update_statement: $ => seq(
            field('left', $.reference),
            field('operator', choice(
                '=',
                '*=',
                '/=',
                '%=',
                '+=',
                '-=',
            )),
            field('right', $._expression),
            $._terminator,
        ),

        select_statement: $ => seq(
            'select',
            '(',
            field('set', $.reference),
            ',',
            field('function', choice($.reference, $.inline_expression_function)),
            ')',
            $._terminator,
        ),

        sample_statement: $ => seq(
            'sample',
            '(',
            field('set', $.reference),
            ',',
            field('parent', $.reference),
            ',',
            field('amount', $._expression),
            ',',
            field('type', choice('ABSOLUTE', 'RELATIVE')),
            ')',
            $._terminator,
        ),

        apply_statement: $ => seq(
            'apply',
            '(',
            field('set', $.reference),
            ',',
            field('function', choice($.reference, $.inline_update_function)),
            ')',
            $._terminator,
        ),

        reduce_statement: $ => seq(
            field('outvar', $.reference),
            '<-',
            'reduce',
            '(',
            field('set', $.reference),
            ',',
            field('function', choice($.reference, $.inline_expression_function)),
            ',',
            field('operator', choice('+', '*',)),
            ')',
            $._terminator,
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
            $.boolean,
            $.unary_expression,
            $.binary_expression,
            $.parenthesized_expression,
            $.reference,
            $.template_variable,
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

        identifier: _ => /[a-zA-Z][_a-zA-Z0-9]*/,

        comment: _ => token(seq('#', /.*/)),

        template_block: _ => token(seq('{%', /[^%]+/, '%}')),

        template_variable: _ => token(seq('{{', /[^}]+/, '}}')),

        _number: $ => choice($.integer, $.float),

        integer: _ => token(repeat1(/[0-9]+/)),

        float: _ => {
            const digits = repeat1(/[0-9]+/);
            const exponent = seq(/[eE][\+-]?/, digits);

            return token(seq(
                choice(
                    seq(digits, '.', optional(digits), optional(exponent)),
                    seq(optional(digits), '.', digits, optional(exponent)),
                    seq(digits, exponent),
                ),
            ));
        },

        boolean: _ => choice('True', 'False'),

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
        $.template_block,
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
