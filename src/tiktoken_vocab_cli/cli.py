import argparse
import csv
import json
import re
import select
import sys
from typing import Dict, Iterable, List, NamedTuple, Tuple

try:
    import tiktoken
except Exception:
    sys.stderr.write(
        "Error: tiktoken is required but not installed.\nInstall with: pip install tiktoken\n"
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
            sys.stderr.write(
                f"Warning: unknown model '{model}'. Falling back to encoding '{encoding or 'cl100k_base'}'.\n"
            )
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
        return all((ch.isprintable() or ch in "\t\n\r ") for ch in s)
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


def build_wrap_pattern(
    left: str, right: str, inner_regex: str | None, ignore_case: bool, greedy: bool
) -> re.Pattern:
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


def build_headers(args: argparse.Namespace) -> List[str]:
    headers = ["token_id", "length", "token_str"]
    if args.show_bytes:
        headers.append("token_bytes_hex")
    if args.show_repr:
        headers.append("token_repr")
    return headers


def collect_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    encoding_name = resolve_encoding_name(args.model, args.encoding)

    regex_obj = build_regex(args.regex, args.ignore_case) if args.regex else None

    wrap_re = None
    if args.wrap_left is not None and args.wrap_right is not None:
        wrap_re = build_wrap_pattern(
            args.wrap_left,
            args.wrap_right,
            args.wrap_inner_regex,
            args.ignore_case,
            args.wrap_greedy,
        )

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
            row: Dict[str, object] = {
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

    return rows


def output_results(args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    headers = build_headers(args)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}", file=sys.stderr)

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

    out = sys.stdout
    if not args.no_header:
        out.write("\t".join(headers) + "\n")
    for r in rows:
        out.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")


def run_cli(args: argparse.Namespace) -> None:
    rows = collect_rows(args)
    output_results(args, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query tiktoken vocabulary with flexible filters.")
    p.add_argument(
        "--interactive", action="store_true", help="Launch interactive menu instead of using flags."
    )
    # Model/encoding
    p.add_argument("--model", help="Model name to resolve encoding (e.g., gpt-4o).")
    p.add_argument(
        "--encoding",
        help="Encoding name (e.g., cl100k_base). Defaults to model-derived or cl100k_base.",
    )
    p.add_argument(
        "--include-special", action="store_true", help="Include special tokens in results."
    )

    # Text filters
    p.add_argument("--regex", help="Regex applied to decoded token string.")
    p.add_argument("--ignore-case", action="store_true", help="Regex ignore case.")
    p.add_argument("--startswith", help="String prefix (decoded).")
    p.add_argument("--endswith", help="String suffix (decoded).")
    p.add_argument("--contains", help="Substring containment (decoded).")

    # Encapsulation / wrapping filters
    p.add_argument("--wrap-left", dest="wrap_left", help="Literal left wrapper, e.g., '<-|'")
    p.add_argument("--wrap-right", dest="wrap_right", help="Literal right wrapper, e.g., '|->'")
    p.add_argument(
        "--wrap-inner-regex",
        dest="wrap_inner_regex",
        help="Regex for inner content between left/right (default: '.+?' non-greedy).",
    )
    p.add_argument(
        "--wrap-greedy", action="store_true", help="Use greedy inner match (default non-greedy)."
    )

    # Repeating nonstandard characters
    p.add_argument(
        "--repeat-nonword-min",
        type=int,
        help="Match tokens containing \u2265N consecutive non-alnum/underscore/non-space characters.",
    )
    p.add_argument(
        "--repeat-chars", help="Literal set of characters to consider for repetition, e.g., '<|-'"
    )
    p.add_argument(
        "--repeat-min",
        type=int,
        default=2,
        help="Minimum run length for --repeat-chars (default: 2).",
    )

    # Byte-level hex filters
    p.add_argument(
        "--bytes-startswith", dest="bytes_startswith", help="Hex prefix, e.g., 'e2 96 81'."
    )
    p.add_argument("--bytes-endswith", dest="bytes_endswith", help="Hex suffix.")
    p.add_argument("--bytes-contains", dest="bytes_contains", help="Hex substring anywhere.")

    # Length & ID filters
    p.add_argument("--min-len", type=int, help="Minimum token length (see --length-by).")
    p.add_argument("--max-len", type=int, help="Maximum token length (see --length-by).")
    p.add_argument(
        "--length-by",
        choices=["chars", "bytes"],
        default="chars",
        help="Count length by characters or bytes.",
    )
    p.add_argument("--min-id", type=int, help="Minimum token id.")
    p.add_argument("--max-id", type=int, help="Maximum token id.")

    # Decoding / printable
    p.add_argument(
        "--decode-errors",
        choices=["replace", "ignore", "strict"],
        default="replace",
        help="How to handle invalid UTF-8 when decoding tokens (default: replace).",
    )
    p.add_argument(
        "--printable",
        action="store_true",
        help="Only include tokens whose decoded form is printable.",
    )
    p.add_argument(
        "--allow-whitespace",
        action="store_true",
        help="When --printable, allow whitespace characters too.",
    )

    # Output / formatting
    p.add_argument(
        "--limit",
        type=int,
        help="Limit number of rows displayed (applied after filtering & sorting).",
    )
    p.add_argument("--sort-by", choices=["id", "len", "token"], default="id", help="Sort results.")
    p.add_argument("--desc", action="store_true", help="Sort descending.")
    p.add_argument("--csv", metavar="PATH", help="Write results to CSV at PATH.")
    p.add_argument("--no-header", action="store_true", help="Do not print header row to stdout.")
    p.add_argument(
        "--show-bytes", action="store_true", help="Include token bytes as hex in output."
    )
    p.add_argument(
        "--show-repr",
        action="store_true",
        help="Include python-style escaped representation of token.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as a JSON array to stdout (suppresses table).",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="Output results as JSON Lines to stdout (suppresses table).",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON with indentation.")

    return p


def prompt_str(prompt: str, current: str | None) -> str | None:
    default = f" [{current}]" if current else ""
    raw = input(f"{prompt}{default} (Enter to keep, '-' to clear): ").strip()
    if not raw:
        return current
    if raw == "-":
        return None
    return raw


def prompt_int(prompt: str, current: int | None) -> int | None:
    default = f" [{current}]" if current is not None else " [None]"
    while True:
        raw = input(f"{prompt}{default} (Enter to keep, '-' to clear): ").strip()
        if not raw:
            return current
        if raw == "-":
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def prompt_bool(prompt: str, current: bool) -> bool:
    options = [
        MenuOption("true", "Yes", "y"),
        MenuOption("false", "No", "n"),
    ]
    choice = select_menu_option(
        f"{prompt} (current: {bool_label(current)})",
        options,
        default_value="true" if current else "false",
        instructions="Use ↑/↓ or listed keys. Enter keeps the highlighted choice.",
    )
    return choice == "true"


def prompt_length_by(current: str) -> str:
    options = [
        MenuOption("chars", "Characters (decoded text)", "1", "Length is measured after decoding."),
        MenuOption("bytes", "Bytes (raw token bytes)", "2", "Length counts the raw byte sequence."),
    ]
    default = current if current in {"chars", "bytes"} else "chars"
    return select_menu_option(
        f"Length basis (current: {default})",
        options,
        default_value=default,
    )


def prompt_sort_by(current: str) -> str:
    options = [
        MenuOption("id", "Token id", "1"),
        MenuOption("len", "Token length", "2"),
        MenuOption("token", "Token string", "3"),
    ]
    default = current if current in {"id", "len", "token"} else "id"
    return select_menu_option(
        f"Sort key (current: {default})",
        options,
        default_value=default,
    )


def prompt_decode_errors(current: str) -> str:
    options = [
        MenuOption("replace", "Replace (�) invalid bytes", "1"),
        MenuOption("ignore", "Ignore invalid bytes", "2"),
        MenuOption("strict", "Strict (raise errors)", "3"),
    ]
    default = current if current in {"replace", "ignore", "strict"} else "replace"
    return select_menu_option(
        f"Decode error handling (current: {default})",
        options,
        default_value=default,
    )


def bool_label(value: bool) -> str:
    return "yes" if value else "no"


class MenuOption(NamedTuple):
    value: str
    label: str
    hotkey: str | None = None
    hint: str | None = None


def _supports_key_capture() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_keypress() -> str | None:
    if not _supports_key_capture():
        return None

    try:
        import termios  # type: ignore[attr-defined]
        import tty  # type: ignore[attr-defined]
    except ImportError:
        try:
            import msvcrt  # type: ignore[attr-defined]
        except ImportError:
            return None
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch += msvcrt.getwch()
        return ch

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            try:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    ch += sys.stdin.read(1)
                    if select.select([sys.stdin], [], [], 0.002)[0]:
                        ch += sys.stdin.read(1)
            except (OSError, ValueError):
                pass
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _normalize_keypress(raw: str) -> str:
    mapping = {
        "\x1b[A": "UP",
        "\x1b[B": "DOWN",
        "\x1b[C": "RIGHT",
        "\x1b[D": "LEFT",
        "\x1bOA": "UP",
        "\x1bOB": "DOWN",
        "\x1bOC": "RIGHT",
        "\x1bOD": "LEFT",
        "\x00H": "UP",
        "\xe0H": "UP",
        "\x00P": "DOWN",
        "\xe0P": "DOWN",
        "\x00M": "RIGHT",
        "\xe0M": "RIGHT",
        "\x00K": "LEFT",
        "\xe0K": "LEFT",
        "\r": "ENTER",
        "\n": "ENTER",
        "\x03": "CTRL_C",
        "\x04": "CTRL_D",
        "\x7f": "BACKSPACE",
        "\x1b": "ESC",
    }
    return mapping.get(raw, raw)


def _match_option_by_key(key: str, options: List[MenuOption]) -> int | None:
    lowered = key.lower()
    for idx, opt in enumerate(options):
        candidates = [opt.value.lower()]
        if opt.hotkey:
            candidates.append(opt.hotkey.lower())
        if lowered in candidates:
            return idx
    return None


def _render_interactive_menu(
    prompt: str,
    options: List[MenuOption],
    highlight_idx: int,
    show_hints: bool,
    instructions: str,
    rendered_lines: int,
) -> int:
    lines = [prompt]
    if instructions:
        lines.append(instructions)
    for idx, opt in enumerate(options):
        pointer = ">" if idx == highlight_idx else " "
        hotkey = f"[{opt.hotkey}] " if opt.hotkey else ""
        lines.append(f"{pointer} {hotkey}{opt.label}")
        if show_hints and idx == highlight_idx and opt.hint:
            lines.append(f"    {opt.hint}")

    if rendered_lines:
        sys.stdout.write(f"\x1b[{rendered_lines}F")
        sys.stdout.write("\x1b[J")
    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return len(lines)


def select_menu_option(
    prompt: str,
    options: List[MenuOption],
    *,
    default_value: str | None = None,
    instructions: str | None = None,
) -> str:
    if not options:
        raise ValueError("options list cannot be empty")

    instructions = (
        instructions
        or "Use ↑/↓ to navigate, Enter to select, listed hotkeys to jump, ? to toggle hints."
    )

    highlight_idx = 0
    if default_value is not None:
        for idx, opt in enumerate(options):
            if opt.value == default_value:
                highlight_idx = idx
                break

    def fallback() -> str:
        print(prompt)
        if instructions:
            print(instructions)
        for opt in options:
            hotkey = f"[{opt.hotkey}] " if opt.hotkey else ""
            print(f"  {hotkey}{opt.label}")
        while True:
            raw = input("Choice: ").strip()
            if not raw and default_value is not None:
                return default_value
            match = _match_option_by_key(raw, options)
            if match is not None:
                return options[match].value
            print("Please select one of the listed options.")

    if not _supports_key_capture():
        return fallback()

    rendered_lines = 0
    show_hints = True
    try:
        sys.stdout.write("\x1b[?25l")
    except Exception:
        pass

    try:
        rendered_lines = _render_interactive_menu(
            prompt, options, highlight_idx, show_hints, instructions, rendered_lines
        )
        while True:
            raw = _read_keypress()
            if raw is None:
                return fallback()
            key = _normalize_keypress(raw)
            if key == "CTRL_C":
                raise KeyboardInterrupt
            if key == "ESC":
                return default_value if default_value is not None else options[highlight_idx].value
            if key == "UP":
                highlight_idx = (highlight_idx - 1) % len(options)
                rendered_lines = _render_interactive_menu(
                    prompt, options, highlight_idx, show_hints, instructions, rendered_lines
                )
                continue
            if key == "DOWN":
                highlight_idx = (highlight_idx + 1) % len(options)
                rendered_lines = _render_interactive_menu(
                    prompt, options, highlight_idx, show_hints, instructions, rendered_lines
                )
                continue
            if key in {"?", "h", "H"}:
                show_hints = not show_hints
                rendered_lines = _render_interactive_menu(
                    prompt, options, highlight_idx, show_hints, instructions, rendered_lines
                )
                continue
            if key == "ENTER":
                sys.stdout.write("\x1b[J")
                sys.stdout.write("\n")
                return options[highlight_idx].value

            match = _match_option_by_key(key, options)
            if match is not None:
                highlight_idx = match
                rendered_lines = _render_interactive_menu(
                    prompt, options, highlight_idx, show_hints, instructions, rendered_lines
                )
                sys.stdout.write("\x1b[J")
                sys.stdout.write("\n")
                return options[highlight_idx].value
    finally:
        try:
            sys.stdout.write("\x1b[?25h")
        except Exception:
            pass

    return fallback()


def run_interactive(
    parser: argparse.ArgumentParser, initial_args: argparse.Namespace | None = None
) -> None:
    base_ns = initial_args or parser.parse_args([])
    config = dict(vars(base_ns))

    if config.get("repeat_min") is None:
        config["repeat_min"] = 2

    menu_options = [
        MenuOption(
            "1", "Configure model & encoding", "1", "Set model/encoding and toggle special tokens."
        ),
        MenuOption(
            "2", "Configure text filters", "2", "Apply regex, prefix/suffix, and substring filters."
        ),
        MenuOption(
            "3",
            "Configure printable & decoding options",
            "3",
            "Control printable filter and decode behaviour.",
        ),
        MenuOption("4", "Configure length & ID filters", "4", "Limit token length or id ranges."),
        MenuOption(
            "5", "Configure byte-level filters", "5", "Filter based on the hex representation."
        ),
        MenuOption(
            "6", "Configure wrapping filters", "6", "Match tokens enclosed by literal wrappers."
        ),
        MenuOption(
            "7",
            "Configure repetition filters",
            "7",
            "Match repeated characters or non-word sequences.",
        ),
        MenuOption(
            "8", "Configure output options", "8", "Adjust limit, sorting, and output formats."
        ),
        MenuOption(
            "9", "Run query with current settings", "9", "Execute the query and show results."
        ),
        MenuOption(
            "0", "Reset all settings to defaults", "0", "Restore parser defaults from argparse."
        ),
        MenuOption("x", "Exit interactive mode", "x", "Close the menu and return to the shell."),
    ]
    selected_choice = "1"

    while True:
        print("\n=== TikToken Vocabulary Explorer ===")
        print(
            f" Model: {config.get('model') or '-'} | Encoding: {config.get('encoding') or 'auto'} | Include special: {bool_label(bool(config.get('include_special', False)))}"
        )
        regex_desc = config.get("regex") or "-"
        print(
            f" Regex: {regex_desc} | Ignore case: {bool_label(bool(config.get('ignore_case', False)))}"
        )
        print(
            f" Printable only: {bool_label(bool(config.get('printable', False)))} | Allow whitespace: {bool_label(bool(config.get('allow_whitespace', False)))} | Decode errors: {config.get('decode_errors', 'replace')}"
        )
        print(
            f" Length range: {config.get('min_len') or '-'} to {config.get('max_len') or '-'} ({config.get('length_by', 'chars')}) | ID range: {config.get('min_id') or '-'} to {config.get('max_id') or '-'}"
        )
        print(
            f" Output: sort by {config.get('sort_by', 'id')} ({'desc' if config.get('desc') else 'asc'}), limit {config.get('limit') or '-'}"
        )

        choice = select_menu_option(
            "Choose an action:",
            menu_options,
            default_value=selected_choice,
        )
        selected_choice = choice

        if choice == "1":
            config["model"] = prompt_str("Model name", config.get("model"))
            config["encoding"] = prompt_str("Encoding name", config.get("encoding"))
            config["include_special"] = prompt_bool(
                "Include special tokens?", bool(config.get("include_special", False))
            )
        elif choice == "2":
            config["regex"] = prompt_str("Regex pattern", config.get("regex"))
            if config.get("regex"):
                config["ignore_case"] = prompt_bool(
                    "Ignore case for regex?", bool(config.get("ignore_case", False))
                )
            else:
                config["ignore_case"] = False
            config["startswith"] = prompt_str("Startswith text", config.get("startswith"))
            config["endswith"] = prompt_str("Endswith text", config.get("endswith"))
            config["contains"] = prompt_str("Contains text", config.get("contains"))
        elif choice == "3":
            config["printable"] = prompt_bool(
                "Require printable tokens?",
                bool(config.get("printable", False)),
            )
            if config["printable"]:
                config["allow_whitespace"] = prompt_bool(
                    "Allow whitespace characters?",
                    bool(config.get("allow_whitespace", False)),
                )
            else:
                config["allow_whitespace"] = bool(config.get("allow_whitespace", False))
            config["decode_errors"] = prompt_decode_errors(config.get("decode_errors", "replace"))
        elif choice == "4":
            config["min_len"] = prompt_int("Minimum length", config.get("min_len"))
            config["max_len"] = prompt_int("Maximum length", config.get("max_len"))
            config["length_by"] = prompt_length_by(config.get("length_by", "chars"))
            config["min_id"] = prompt_int("Minimum token ID", config.get("min_id"))
            config["max_id"] = prompt_int("Maximum token ID", config.get("max_id"))
        elif choice == "5":
            config["bytes_startswith"] = prompt_str(
                "Hex prefix (bytes startswith)", config.get("bytes_startswith")
            )
            config["bytes_endswith"] = prompt_str(
                "Hex suffix (bytes endswith)", config.get("bytes_endswith")
            )
            config["bytes_contains"] = prompt_str(
                "Hex substring (bytes contains)", config.get("bytes_contains")
            )
        elif choice == "6":
            config["wrap_left"] = prompt_str("Wrap left literal", config.get("wrap_left"))
            config["wrap_right"] = prompt_str("Wrap right literal", config.get("wrap_right"))
            config["wrap_inner_regex"] = prompt_str(
                "Inner wrap regex", config.get("wrap_inner_regex")
            )
            config["wrap_greedy"] = prompt_bool(
                "Use greedy inner wrap match?",
                bool(config.get("wrap_greedy", False)),
            )
        elif choice == "7":
            config["repeat_nonword_min"] = prompt_int(
                "Min consecutive non-word characters",
                config.get("repeat_nonword_min"),
            )
            config["repeat_chars"] = prompt_str(
                "Character set for repetition", config.get("repeat_chars")
            )
            if config["repeat_chars"]:
                repeat_min = prompt_int("Minimum run length", config.get("repeat_min", 2))
                config["repeat_min"] = repeat_min if repeat_min is not None else 2
            else:
                config["repeat_min"] = 2
        elif choice == "8":
            config["limit"] = prompt_int("Row limit", config.get("limit"))
            config["sort_by"] = prompt_sort_by(config.get("sort_by", "id"))
            config["desc"] = prompt_bool("Sort descending?", bool(config.get("desc", False)))
            config["show_bytes"] = prompt_bool(
                "Include bytes column?", bool(config.get("show_bytes", False))
            )
            config["show_repr"] = prompt_bool(
                "Include repr column?", bool(config.get("show_repr", False))
            )
            config["csv"] = prompt_str("CSV output path", config.get("csv"))
            config["json"] = prompt_bool("Output JSON array?", bool(config.get("json", False)))
            if config["json"]:
                config["pretty"] = prompt_bool(
                    "Pretty print JSON?", bool(config.get("pretty", False))
                )
                config["jsonl"] = False
            else:
                config["jsonl"] = prompt_bool(
                    "Output JSON Lines?", bool(config.get("jsonl", False))
                )
                if not config["jsonl"]:
                    config["pretty"] = bool(config.get("pretty", False))
            config["no_header"] = prompt_bool(
                "Skip header row?", bool(config.get("no_header", False))
            )
        elif choice == "9":
            run_args = dict(config)
            run_args["interactive"] = False
            try:
                run_cli(argparse.Namespace(**run_args))
            except KeyboardInterrupt:
                print("\nQuery cancelled.")
            except Exception as exc:
                print(f"\nError running query: {exc}")
            input("\nPress Enter to return to the menu...")
        elif choice == "0":
            config = dict(vars(parser.parse_args([])))
            config["repeat_min"] = config.get("repeat_min", 2) or 2
            selected_choice = "1"
        elif choice == "x":
            print("Exiting interactive mode.")
            return
        else:
            print("Unrecognized option.")


def main(argv: List[str] | None = None):
    parser = build_arg_parser()
    provided_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(provided_args)

    if args.interactive or not provided_args:
        run_interactive(parser, args if args.interactive else None)
        return

    run_cli(args)


if __name__ == "__main__":
    main()
