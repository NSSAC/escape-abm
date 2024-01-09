.PHONY : parser test

export PATH := ./node_modules/.bin/:$(PATH)

parser: src/parser.c

src/parser.c: grammar.js
	tree-sitter generate
	rsync -a --del src/ ../src/episim37/tree-sitter-esl37/src/

test: parser
	tree-sitter test

