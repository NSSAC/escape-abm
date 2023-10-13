.PHONY : parser test

export PATH := ./node_modules/.bin/:$(PATH)
PROJECT_ROOT = ..

parser: src/parser.c

src/parser.c : grammar.js
	tree-sitter generate
	cp -a src/parser.c $(PROJECT_ROOT)/src/episim37/tree-sitter-esl37/src

test : parser
	tree-sitter test
