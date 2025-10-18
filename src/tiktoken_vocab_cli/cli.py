import argparse
import csv
import json
import re
import sys
from typing import Dict, Iterable, List, Tuple

try:
    import tiktoken
except Exception:
    sys.stderr.write(
        "Error: tiktoken is required but not installed.\n"
        "Install with: pip install tiktoken\n"
    )
    sys.exit(1)


def resolve_encoding_name(model: str | None, encoding: str | None) -> str:
    """
    Resolve encoding from --model if provided; otherwise fall back to --encoding (or default).
    """
    if model:
        try:
            enc = tiktoken.encoding_for_model(model)
            return enc.name
        except KeyError:
            sys.stderr.write(f"Warning: unknown model '{model}'. Falling back to encoding '{encoding or 'cl100k_base'}'.\n")
    return encoding or "cl100k_base"


def iter_vocab(encoding_name: str, include_special: bool) -> Iterable[Tuple[int, bytes]]:
    enc = tiktoken.get_encoding(encoding_name)
    items = list(enc._mergeable_ranks.items())  # type: ignore[attr-defined]
    if include_special:
        for s, i in enc._special_tokens.items():  # type: ignore[attr-defined]
            items.append((s.encode("utf-8", "replace"), i))
    for b, i in items:
        yield i, b


def is_printable(s: str, allow_whitespace: bool) -> bool:
    if allow_whitespace:
        return all((ch.isprintable() or ch in "\t\n\r ")) for ch in s)
    return s.isprintable()


def bytes_to_hex(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)


def safe_repr(s: str) -> str:
    return s.encode("unicode_escape").decode("ascii")


def build_regex(pattern: str, ignore_case: bool) -> re.Pattern:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)


def token_len(b: bytes, by: str, decode_errors: str) -> int:
    if by == "chars":
        return len(b.decode("utf-8", errors=decode_errors))
    elif by == "bytes":
        return len(b)
    else:
        raise ValueError("--length-by must be 'chars' or 'bytes'")


def char_class_from_literals(chars: str) -> str:
    if not chars:
        return ""
    escaped = "".join(re.escape(c) for c in chars)
    return f"[{escaped}]"


def build_wrap_pattern(left: str, right: str, inner_regex: str | None, ignore_case: bool, greedy: bool) -> re.Pattern:
    L = re.escape(left)
    R = re.escape(right)
    inner = inner_regex if inner_regex is not None else (".+" if greedy else ".+?")
    pat = f"{L}{inner}{R}"
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pat, flags)


def matches_filters(
    *,
    token_bytes: bytes,
    token_str: str,
    token_id: int,
    args: argparse.Namespace,
    regex_obj: re.Pattern | None,
    wrap_re: re.Pattern | None,
    repeat_nonword_re: re.Pattern | None,
    repeat_chars_re: re.Pattern | None,
) -> bool:

    # Regex on decoded string
    if args.regex:
        if not regex_obj.search(token_str):
            return False

    # startswith / endswith / contains on decoded string
    if args.startswith and not token_str.startswith(args.startswith):
        return False
    if args.endswith and not token_str.endswith(args.endswith):
        return False
    if args.contains and args.contains not in token_str:
        return False

    # bytes hex prefix/suffix/contains
    if args.bytes_startswith or args.bytes_endswith or args.bytes_contains:
        hex_str = bytes_to_hex(token_bytes)
        if args.bytes_startswith and not hex_str.startswith(args.bytes_startswith.lower()):
            return False
        if args.bytes_endswith and not hex_str.endswith(args.bytes_endswith.lower()):
            return False
        if args.bytes_contains and args.bytes_contains.lower() not in hex_str:
            return False

    # printable
    if args.printable and not is_printable(token_str, args.allow_whitespace):
        return False

    # length filters
    length_val = token_len(token_bytes, args.length_by, args.decode_errors)
    if args.min_len is not None and length_val < args.min_len:
        return False
    if args.max_len is not None and length_val > args.max_len:
        return False

    # id range
    if args.min_id is not None and token_id < args.min_id:
        return False
    if args.max_id is not None and token_id > args.max_id:
        return False

    # wrapping pattern (encapsulation)
    if wrap_re and not wrap_re.search(token_str):
        return False

    # repeating nonword chars (not alnum/underscore/space)
    if repeat_nonword_re and not repeat_nonword_re.search(token_str):
        return False

    # repeating from a custom set
    if repeat_chars_re and not repeat_chars_re.search(token_str):
        return False

    return True


def sort_key(row: Dict[str, object], key: str):
    if key == "id":
        return int(row["token_id"])  # type: ignore[index]
    if key == "len":
        return int(row["length"])  # type: ignore[index]
    if key == "token":
        return str(row["token_str"])  # type: ignore[index]
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query tiktoken vocabulary with flexible filters.")
    # Model/encoding
    p.add_argument("--model", help="Model name to resolve encoding (e.g., gpt-4o).")
    p.add_argument("--encoding", help="Encoding name (e.g., cl100k_base). Defaults to model-derived or cl100k_base.")
    p.add_argument("--include-special", action="store_true", help="Include special tokens in results.")

    # Text filters
    p.add_argument("--regex", help="Regex applied to decoded token string.")
    p.add_argument("--ignore-case", action="store_true", help="Regex ignore case.")
    p.add_argument("--startswith", help="String prefix (decoded).")
    p.add_argument("--endswith", help="String suffix (decoded).")
    p.add_argument("--contains", help="Substring containment (decoded).")

    # Encapsulation / wrapping filters
    p.add_argument("--wrap-left", dest="wrap_left", help="Literal left wrapper, e.g., '<-|'")
    p.add_argument("--wrap-right", dest="wrap_right", help="Literal right wrapper, e.g., '|->'")
    p.add_argument("--wrap-inner-regex", dest="wrap_inner_regex",
                   help="Regex for inner content between left/right (default: '.+?' non-greedy).")
    p.add_argument("--wrap-greedy", action="store_true", help="Use greedy inner match (default non-greedy).")

    # Repeating nonstandard characters
    p.add_argument("--repeat-nonword-min", type=int,
                   help="Match tokens containing \u2265N consecutive non-alnum/underscore/non-space characters.")
    p.add_argument("--repeat-chars", help="Literal set of characters to consider for repetition, e.g., '<|-'")
    p.add_argument("--repeat-min", type=int, default=2, help="Minimum run length for --repeat-chars (default: 2).")

    # Byte-level hex filters
    p.add_argument("--bytes-startswith", dest="bytes_startswith", help="Hex prefix, e.g., 'e2 96 81'.")
    p.add_argument("--bytes-endswith", dest="bytes_endswith", help="Hex suffix.")
    p.add_argument("--bytes-contains", dest="bytes_contains", help="Hex substring anywhere.")

    # Length & ID filters
    p.add_argument("--min-len", type=int, help="Minimum token length (see --length-by).")
    p.add_argument("--max-len", type=int, help="Maximum token length (see --length-by).")
    p.add_argument("--length-by", choices=["chars", "bytes"], default="chars", help="Count length by characters or bytes.")
    p.add_argument("--min-id", type=int, help="Minimum token id.")
    p.add_argument("--max-id", type=int, help="Maximum token id.")

    # Decoding / printable
    p.add_argument("--decode-errors", choices=["replace", "ignore", "strict"], default="replace",
                   help="How to handle invalid UTF-8 when decoding tokens (default: replace).")
    p.add_argument("--printable", action="store_true", help="Only include tokens whose decoded form is printable.")
    p.add_argument("--allow-whitespace", action="store_true", help="When --printable, allow whitespace characters too.")

    # Output / formatting
    p.add_argument("--limit", type=int, help="Limit number of rows displayed (applied after filtering & sorting).")
    p.add_argument("--sort-by", choices=["id", "len", "token"], default="id", help="Sort results.")
    p.add_argument("--desc", action="store_true", help="Sort descending.")
    p.add_argument("--csv", metavar="PATH", help="Write results to CSV at PATH.")
    p.add_argument("--no-header", action="store_true", help="Do not print header row to stdout.")
    p.add_argument("--show-bytes", action="store_true", help="Include token bytes as hex in output.")
    p.add_argument("--show-repr", action="store_true", help="Include python-style escaped representation of token.")
    p.add_argument("--json", action="store_true", help="Output results as a JSON array to stdout (suppresses table).")
    p.add_argument("--jsonl", action="store_true", help="Output results as JSON Lines to stdout (suppresses table).")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON with indentation.")

    return p


def main(argv: List[str] | None = None):
    p = build_arg_parser()
    args = p.parse_args(argv)

    encoding_name = resolve_encoding_name(args.model, args.encoding)

    regex_obj = build_regex(args.regex, args.ignore_case) if args.regex else None

    wrap_re = None
    if args.wrap_left is not None and args.wrap_right is not None:
        wrap_re = build_wrap_pattern(args.wrap_left, args.wrap_right, args.wrap_inner_regex,
                                     args.ignore_case, args.wrap_greedy)

    repeat_nonword_re = None
    if args.repeat_nonword_min:
        n = args.repeat_nonword_min
        repeat_nonword_re = re.compile(rf"[^\w\s]{{{n},}}")

    repeat_chars_re = None
    if args.repeat_chars:
        cls = char_class_from_literals(args.repeat_chars)
        repeat_chars_re = re.compile(rf"{cls}{{{args.repeat_min},}}")

    rows: List[Dict[str, object]] = []
    for token_id, token_bytes in iter_vocab(encoding_name, args.include_special):
        try:
            token_str = token_bytes.decode("utf-8", errors=args.decode_errors)
        except Exception:
            continue

        if matches_filters(
            token_bytes=token_bytes,
            token_str=token_str,
            token_id=token_id,
            args=args,
            regex_obj=regex_obj,
            wrap_re=wrap_re,
            repeat_nonword_re=repeat_nonword_re,
            repeat_chars_re=repeat_chars_re,
        ):
            row = {
                "token_id": token_id,
                "length": token_len(token_bytes, args.length_by, args.decode_errors),
                "token_str": token_str,
            }
            if args.show_bytes:
                row["token_bytes_hex"] = bytes_to_hex(token_bytes)
            if args.show_repr:
                row["token_repr"] = safe_repr(token_str)
            rows.append(row)

    rows.sort(key=lambda r: sort_key(r, args.sort_by), reverse=args.desc)

    if args.limit is not None:
        rows = rows[: args.limit]

    headers = ["token_id", "length", "token_str"]
    if args.show_bytes:
        headers.append("token_bytes_hex")
    if args.show_repr:
        headers.append("token_repr")

    # CSV output (optional)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}", file=sys.stderr)

    # JSON / JSONL output modes (suppress table)
    if args.json or args.jsonl:
        if args.jsonl:
            for r in rows:
                print(json.dumps(r, ensure_ascii=False))
        else:
            if args.pretty:
                print(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(rows, ensure_ascii=False))
        return

    # Stdout table (TSV-like)
    out = sys.stdout
    if not args.no_header:
        out.write("\t".join(headers) + "\n")
    for r in rows:
        out.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")


if __name__ == "__main__":
    main()
