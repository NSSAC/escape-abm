.PHONY : parser test

test:
	make -C tree-sitter-esl37 test
	make -C tests/examples all

parser:
	make -C tree-sitter-esl37 parser


