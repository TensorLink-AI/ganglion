# Tool Validation Rules

Extracted from `src/ganglion/state/validator.py` (`MutationValidator`), extended
for the MCP plugin context.

## Blocked imports

Tools submitted via `ganglion_publish_tool` are rejected if they contain any of
these imports:

### Original (from validator.py)
- `subprocess`
- `os.system`
- `shutil.rmtree`
- `socket`
- `http.server`

### Added for MCP
- `eval`
- `exec`
- `__import__`

The validator checks both `import X` and `from X import ...` forms.  It matches
when the full import name starts with a blocked entry or when the module name
starts with a blocked entry.

## Structural checks

1. **`@tool` decorator required** — the code must contain at least one function
   decorated with `@tool`.
2. **Type hints on all parameters** — every parameter of the decorated function
   must have a type annotation.
3. **Docstring required** — the decorated function must have a docstring.
4. **Valid Python syntax** — the code must parse without `SyntaxError`.

## Size limits

- **Max code size: 50 KB** per tool submission.

## Implementation notes

The validator uses Python's `ast` module to parse the code and walk the AST:
- `ast.parse(code)` for syntax validation
- Walk `ast.Import` and `ast.ImportFrom` nodes for blocked import detection
- Check `ast.FunctionDef` nodes for decorator, annotation, and docstring presence
