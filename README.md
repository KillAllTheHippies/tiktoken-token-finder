
## tiktoken-vocab-cli

Command line tooling for exploring OpenAI `tiktoken` vocabularies. The CLI runs either through direct flags or an interactive menu that guides configuration of filters, output formats, and token metadata.

---

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt  # install runtime deps (includes tiktoken)
pip install -r requirements-dev.txt  # optional: dev tooling (ruff, black, pytest)
```

`tiktoken>=0.5` is required. The CLI exits early with a clear error if the package is missing.

---

### Quick Start

```bash
# List tokens starting with "the"
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --regex "^the"

# Launch the interactive menu (recommended for discovery)
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --interactive
```

Running the module without arguments drops directly into interactive mode.

---

### Interactive Menu Overview

Navigation uses the arrow keys or hotkeys shown next to each option:

- Arrow up/down: move selection.
- `Enter`: apply highlighted option.
- Hotkey (numbers or listed letters): jump directly to an action.
- `?`: toggle inline hints.
- `x` or `q`: exit interactive mode.

State is displayed at the top of the menu, reflecting the effective model, encoding, filters, and output preferences. Picking a model automatically sets the matching encoding; overrides remain possible afterward.

Key menu actions:

1. **Configure model & encoding**
   - Choose a known model (populated from `tiktoken.model.MODEL_TO_ENCODING`) or specify a custom name.
   - Encoding resolves automatically from the model; you can replace it manually (from `tiktoken.list_encoding_names`).
   - Toggle inclusion of special tokens or restrict results to specials only.

2. **Configure text filters**
   - Regex (Python syntax) with quick-reference hints.
   - Plain string filters: startswith, endswith, contains.
   - Case-insensitive flag for regex.

3. **Printable & decoding options**
   - Require printable output, optionally allowing whitespace.
   - Choose UTF-8 error handling (`replace`, `ignore`, `strict`).

4. **Length & ID filters**
   - Length bounds counted by characters or bytes.
   - Minimum and maximum token IDs.
   - Explicit token ID list or ranges (e.g., `199999, 200000-200010`).
   - Range filters and explicit IDs combine by intersection: if IDs are listed, only those IDs remain even if they fall outside range limits.

5. **Byte-level filters**
   - Hex-based prefix, suffix, or contains filters against the raw token bytes.

6. **Wrapping filters**
   - Require literal left and right markers, optionally with a custom regex for the inner content.
   - Greedy/non-greedy inner match toggle.

7. **Repetition filters**
   - Minimum run of non-word characters.
   - Custom character set + minimum repetition length.

8. **Output options**
   - Limit rows, set sort column and direction.
   - Toggle bytes and repr columns.
   - Select CSV path or JSON/JSONL output (mutually exclusive).
   - When JSON is enabled, `--pretty` style indentation is available.

9. **Run query**
   - Executes with current settings; errors and cancellations are surfaced directly.
   - After completion, press Enter to return to the menu.

0. **Reset settings**
   - Restores argparse defaults, including repr and length settings.

`x` / `q`. **Exit interactive mode.**

---

### Flag-Based Usage

All interactive capabilities are mirrored by command-line flags:

```bash
PYTHONPATH=src python -m tiktoken_vocab_cli.cli \
  --model gpt-4o-mini \
  --regex "^the" \
  --token-id 200000-200010 \
  --printable --allow-whitespace \
  --length-by chars --min-len 3 \
  --sort-by token --desc \
  --limit 25
```

Common flag notes:

- `--model` resolves an encoding automatically. If the model name is unknown, it falls back to `--encoding` or `cl100k_base`.
- `--encoding` can always be set explicitly.
- `--token-id` accepts integers or ranges; provide multiple flags to accumulate more ranges.
- `--special-only` is shorthand for `--include-special --special-only`.

---

### Output Formats

- **Tabular (default):** TSV with headers unless `--no-header`.
- **CSV:** `--csv path.csv` writes to disk and still prints to stdout unless `--no-header`.
- **JSON:** `--json` emits an array; `--pretty` indents it.
- **JSONL:** `--jsonl` prints one object per line. JSON and JSONL suppress the table.
- **Additional columns:** `--show-bytes` for hex bytes; `--show-repr` for Python repr.

Rows include:

| Column            | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `token_id`        | Integer identifier in the encoding (includes specials).  |
| `length`          | Token length in characters or bytes depending on `--length-by`. |
| `token_str`       | UTF-8 decoded string using the configured error strategy. |
| `token_bytes_hex` | Optional when `--show-bytes`.                             |
| `token_repr`      | Optional when `--show-repr`; uses `repr()` semantics.     |

---

### Special Tokens and Explicit IDs

- `--include-special` merges special tokens into the vocabulary; `--special-only` restricts matches to that set.
- Explicit token IDs force `include_special=True` automatically to guarantee the IDs appear, even when they belong to the special token table.
- Token ID ranges are inclusive on both ends; range start must be less than or equal to end.

---

### Error Handling and Edge Cases

- Regex compilation failures surface immediately with Python’s `re` error.
- Non-existent encoding or model names fallback to defaults with a warning.
- Unsupported token IDs simply filter to an empty result set; no error is raised.

---

### Development and Checks

```bash
./.venv/bin/ruff check src
./.venv/bin/ruff format src
./.venv/bin/black src
pytest  # once tests are present under tests/
```

Use `PYTHONPATH=src` so imports resolve to the in-repo package. The CLI has no server component or persistent state—every invocation rebuilds filters from the arguments.

---

### Useful Recipes

```bash
# Show the highest token IDs (specials included)
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --include-special --sort-by id --desc --limit 20

# Compare printable vs non-printable tokens
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --printable --limit 10
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --printable --allow-whitespace --limit 10

# Export to CSV
PYTHONPATH=src python -m tiktoken_vocab_cli.cli --regex "\\d+" --csv digits.csv --limit 200
```

No external services are used: the CLI relies solely on local `tiktoken` vocabularies. The tool is deterministic—the same set of filters over the same encoding produces the same output each run.
