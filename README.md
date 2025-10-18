

### JSON output

```bash
# JSON array
tiktoken-vocab --regex "^[A-Z]{2,}$" --json --pretty > tokens.json

# JSON Lines
tiktoken-vocab --repeat-nonword-min 3 --jsonl > weird_tokens.jsonl
```

### Resolve encoding from a model

```bash
tiktoken-vocab --model gpt-4o-mini --regex "^the"
```
