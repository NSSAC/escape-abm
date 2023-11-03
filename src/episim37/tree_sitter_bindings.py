"""Python bindings for the tree-sitter-parser."""


from pathlib import Path
from tree_sitter import Language, Parser

from platformdirs import user_cache_dir

LANGUAGE = "esl37"
TREE_SITTER_DIR = Path(__file__).parent / f"tree-sitter-{LANGUAGE}"


def get_parser() -> Parser:
    state_dir = user_cache_dir(appname=LANGUAGE)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    library_file = state_dir / "languages.so"

    Language.build_library(str(library_file), [str(TREE_SITTER_DIR)])
    language = Language(str(library_file), LANGUAGE)

    parser = Parser()
    parser.set_language(language)

    return parser
