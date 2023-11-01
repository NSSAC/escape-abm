.PHONY : parser test

export PATH := ./node_modules/.bin/:$(PATH)

parser: src/parser.c

src/parser.c: grammar.js
	tree-sitter generate

test: parser
	tree-sitter test

