import re
from typing import Optional, Tuple
from decimal import Decimal, InvalidOperation
import math
import ast
import operator
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
_SYMPY_AVAILABLE = True
_SYMPY_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


_BOX_TOKENS = (r"\boxed", r"\fbox")

_CURRENCY_SYMBOLS = r"$£€¥₩₹₽₺₫฿₦₴₲₡₱¢"
_CURRENCY_WORDS = [
    "usd", "us$", "dollar", "dollars",
    "eur", "euro", "euros",
    "gbp", "pound", "pounds",
    "jpy", "yen",
    "krw", "won",
    "inr", "rupee", "rupees",
    "cad", "aud", "chf", "cny", "rmb",
]

_UNITS_NO_SPACE = [
    "lb", "lbs", "kg", "mg", "oz",
    "cm", "mm", "km",
    "mph", "kph",
    "ft", "yd", "in", "mi",
    "ml", "l", "gal",
]

_UNITS_SPACE_ONLY = [
    "m", "s", "g",
]

_UNIT_WORDS = [
    "pound", "pounds", "ounce", "ounces",
    "meter", "meters", "metre", "metres",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "mile", "miles", "yard", "yards", "inch", "inches", "foot", "feet",
    "liter", "liters", "litre", "litres", "gallon", "gallons",
]

_ALLOWED_NUMEXPR_CHARS_RE = re.compile(
    r"^[0-9+\-*/^().,\s]+(?:e[+\-]?\d+)?\s*$",
    re.IGNORECASE,
)

_TEXT_CMD_RE = re.compile(r"\\(?:text|mathrm|mathit|mathbf|mbox)\{([^{}]*)\}")

_LATEX_SPACE_CMDS_RE = re.compile(
    r"\\(?:,|;|:|!|quad|qquad|hspace|vspace)\b(?:\{[^{}]*\})?"
)

_NUM_ONLY_RE = re.compile(
    r"^[+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:e[+\-]?\d+)?$",
    re.IGNORECASE,
)

_CURRENCY_WORDS_ALT = "|".join(map(re.escape, _CURRENCY_WORDS))
_UNITS_NO_SPACE_ALT = "|".join(map(re.escape, _UNITS_NO_SPACE))
_UNITS_SPACE_ONLY_ALT = "|".join(map(re.escape, _UNITS_SPACE_ONLY))
_UNIT_WORDS_ALT = "|".join(map(re.escape, _UNIT_WORDS))


def _extract_balanced_braces(s: str, open_idx: int) -> Tuple[Optional[str], Optional[int]]:
    r"""
    Given s[open_idx] == '{', return (content_inside, index_after_closing_brace).
    Handles nested braces and skips escaped characters like \{, \}, \$.
    """
    if open_idx >= len(s) or s[open_idx] != "{":
        return None, None

    depth = 0
    i = open_idx
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : i], i + 1
        i += 1

    return None, None


def _unescape_latex_specials(s: str) -> str:
    return re.sub(r"\\([$%#&_{}])", r"\1", s)


def _strip_trailing_punct(s: str) -> str:
    return re.sub(r"[ \t]*[.,;:!?]+[ \t]*$", "", s).strip()


def _normalize_degree_and_percent(s: str) -> str:
    s = s.replace("°", r"^\circ")
    s = re.sub(r"\^\{\s*\\circ\s*\}", r"^\\circ", s)
    s = re.sub(r"(\d)\s*\\circ\b", r"\1^\\circ", s)

    s = s.replace(r"\%", "%")
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    return s


def _unwrap_text_like_wrappers(s: str) -> str:
    r"""
    Repeatedly unwrap simple wrappers like \text{...}, \mathrm{...}, \mathbf{...}.
    """
    for _ in range(5):
        s2 = _TEXT_CMD_RE.sub(r"\1", s)
        if s2 == s:
            break
        s = s2
    return s


def _looks_numericish_for_affix_stripping(s: str) -> bool:
    r"""
    Decide whether currency/unit stripping is safe.
    This is intentionally permissive for things like:
      $70000, \$70,000, 23 kg, \frac{1}{2}kg, 90^\circ, 25\%
    """
    gate = s.strip()
    gate = _unescape_latex_specials(gate)
    gate = _unwrap_text_like_wrappers(gate)
    gate = _strip_trailing_punct(gate)

    # Remove obvious currency affixes on a copy.
    gate = re.sub(rf"^\s*[{re.escape(_CURRENCY_SYMBOLS)}]\s*", "", gate)
    gate = re.sub(rf"^\s*(?i:{_CURRENCY_WORDS_ALT})\b\s*", "", gate)
    gate = re.sub(rf"\s*[{re.escape(_CURRENCY_SYMBOLS)}]\s*$", "", gate)
    gate = re.sub(rf"\s*(?i:{_CURRENCY_WORDS_ALT})\b\.?\s*$", "", gate)

    # Remove one layer of trailing unit affixes on a copy.
    if _UNITS_NO_SPACE_ALT:
        gate = re.sub(rf"\s*(?i:{_UNITS_NO_SPACE_ALT})\b\.?\s*$", "", gate)
    if _UNITS_SPACE_ONLY_ALT:
        gate = re.sub(rf"\s+(?i:{_UNITS_SPACE_ONLY_ALT})\b\.?\s*$", "", gate)
    if _UNIT_WORDS_ALT:
        gate = re.sub(rf"\s*(?i:{_UNIT_WORDS_ALT})\b\.?\s*$", "", gate)

    # Ignore percent/degree markers for the gate.
    gate = gate.replace(r"\%", "").replace("%", "")
    gate = gate.replace(r"^\circ", "").replace("°", "")

    gate_no_spaces = re.sub(r"\s+", "", gate)

    return bool(re.search(r"\d", gate_no_spaces)) and (
        ("\\" in gate_no_spaces) or bool(_ALLOWED_NUMEXPR_CHARS_RE.match(gate))
    )


def _extract_boxed_fallback_token(s: str) -> Optional[str]:
    r"""
    Conservative fallback for malformed outputs like: \boxed 23
    Stops at whitespace/newline. Trailing punctuation is cleaned later.
    """
    m = re.match(r"\s*([^\s\n\r]+)", s)
    if not m:
        return None
    cand = m.group(1).strip()
    return cand or None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last boxed content from text using brace parsing.
    Works for arbitrarily nested braces.
    """
    if not text:
        return None

    end = len(text)
    while end > 0:
        last_pos = -1
        for tok in _BOX_TOKENS:
            p = text.rfind(tok, 0, end)
            if p > last_pos:
                last_pos = p

        if last_pos == -1:
            return None

        pos = last_pos
        matched_tok = None
        for tok in _BOX_TOKENS:
            if text.startswith(tok, last_pos):
                matched_tok = tok
                break

        if matched_tok is None:
            end = last_pos
            continue

        pos += len(matched_tok)

        while pos < len(text) and text[pos].isspace():
            pos += 1

        if pos < len(text) and text[pos] == "{":
            content, _ = _extract_balanced_braces(text, pos)
            if content is not None:
                content = content.strip()
                if content:
                    return content

        cand = _extract_boxed_fallback_token(text[pos:])
        if cand:
            return cand

        end = last_pos

    return None


def clean_extracted_answer(
    ans: str,
    *,
    strip_currency: bool = False,
    strip_units: bool = False,
    normalize_degree_percent: bool = True,
) -> str:
    r"""
    Cleans an extracted answer.

    By default this function is conservative:
      - preserves currency/unit surface forms,
      - unescapes LaTeX specials (\$ -> $),
      - unwraps simple LaTeX text wrappers,
      - optionally strips currency/unit affixes only when safe,
      - normalizes degree/percent notation.

    Use strip_currency=True / strip_units=True only for comparison-oriented normalization.
    """
    if ans is None:
        return ""

    s = ans.strip()

    # Remove outer math fences only when they are true fences, e.g. "$x$".
    # Do not treat "$70000" as fenced math.
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$" and s.count("$") >= 2:
        s = s[1:-1].strip()

    s = _unescape_latex_specials(s)
    s = _unwrap_text_like_wrappers(s)
    s = _strip_trailing_punct(s)

    if normalize_degree_percent:
        s = _normalize_degree_and_percent(s)

    looks_numexpr = _looks_numericish_for_affix_stripping(s)

    if looks_numexpr:
        if strip_currency:
            s = re.sub(rf"^\s*[{re.escape(_CURRENCY_SYMBOLS)}]\s*", "", s)
            s = re.sub(rf"^\s*(?i:{_CURRENCY_WORDS_ALT})\b\s*", "", s)

            s = re.sub(rf"\s*[{re.escape(_CURRENCY_SYMBOLS)}]\s*$", "", s)
            s = re.sub(rf"\s*(?i:{_CURRENCY_WORDS_ALT})\b\.?\s*$", "", s)

        if strip_units:
            if _UNITS_NO_SPACE_ALT:
                s = re.sub(rf"\s*(?i:{_UNITS_NO_SPACE_ALT})\b\.?\s*$", "", s)

            if _UNITS_SPACE_ONLY_ALT:
                s = re.sub(rf"\s+(?i:{_UNITS_SPACE_ONLY_ALT})\b\.?\s*$", "", s)

            if _UNIT_WORDS_ALT:
                s = re.sub(rf"\s*(?i:{_UNIT_WORDS_ALT})\b\.?\s*$", "", s)

        s = _strip_trailing_punct(s).strip()

    return s


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    final_section = text
    if "</think>" in text:
        final_section = text.split("</think>", 1)[1].strip()

    search_blobs = [final_section, text] if final_section != text else [text]

    # Boxed answers: preserve the surface form here.
    for blob in search_blobs:
        boxed = extract_boxed_answer(blob)
        if boxed:
            cleaned = clean_extracted_answer(
                boxed,
                strip_currency=False,
                strip_units=False,
                normalize_degree_percent=True,
            )
            if cleaned:
                return cleaned

    final_answer_patterns = [
        r"\bthe\s+final\s+answer\s+is\s*\[([^\[\]\n\r]+)\]",
        r"\bthe\s+final\s+answer\s+is\s*([^\n\r]+)",
        r"\bfinal\s+answer\b\s*[:=]\s*([^\n\r]+)",
        r"\banswer\b\s*[:=]\s*([^\n\r]+)",
        r"\bthus\b.*?\banswer\b\s*[:=]?\s*([^\n\r]+)",
    ]

    for blob in search_blobs:
        for pat in final_answer_patterns:
            m = re.search(pat, blob, flags=re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cand = clean_extracted_answer(
                    cand,
                    strip_currency=False,
                    strip_units=False,
                    normalize_degree_percent=True,
                )
                if cand:
                    return cand

    return None


def _strip_outer_braces_if_balanced(s: str) -> str:
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return s

    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
        i += 1

    return s[1:-1].strip() if depth == 0 else s


def _normalize_frac_sqrt_and_subscript_syntax(s: str) -> str:
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    # 4210_{5} -> 4210_5
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)

    # \frac14 -> \frac{1}{4}
    s = re.sub(r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])", r"\\frac{\1}{\2}", s)

    # \frac{270}7 -> \frac{270}{7}
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*([A-Za-z0-9])", r"\\frac{\1}{\2}", s)

    # \frac1{4} -> \frac{1}{4}
    s = re.sub(r"\\frac\s*([A-Za-z0-9])\s*\{([^{}]+)\}", r"\\frac{\1}{\2}", s)

    # 11\sqrt2 -> 11\sqrt{2}
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])\b", r"\\sqrt{\1}", s)

    return s


def _strip_terminal_ordinal_suffix(s: str) -> str:
    return re.sub(
        r"(?<=\d)\s*(?:\^\{(?:st|nd|rd|th)\}|\^(?:st|nd|rd|th)|(?:st|nd|rd|th))\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()


def _strip_terminal_degree_markers(s: str) -> str:
    return re.sub(r"(?:\^\{\\circ\}|\^\\circ|°)\s*$", "", s).strip()


def _strip_numeric_trailing_units(s: str) -> str:
    x = s
    x = re.sub(
        r"\s*(?:inch|inches|degree|degrees)\b(?:\s*\^\{?\d+\}?)?\s*$",
        "",
        x,
        flags=re.IGNORECASE,
    )
    return x.strip()


def _normalize_plain_text_answer(s: str) -> Optional[str]:
    t = s.strip()
    t = _strip_outer_braces_if_balanced(t)
    t = t.strip()

    m = re.fullmatch(r"\(\s*([A-Za-z])\s*\)", t)
    if m:
        return m.group(1).lower()

    if re.fullmatch(r"[A-Za-z]+", t):
        return t.lower()

    return None


def _latex_to_python_expr_general(s: str) -> Optional[str]:
    if not s:
        return None

    s = _strip_latex_noise(s)
    s = _normalize_frac_sqrt_and_subscript_syntax(s)
    s = _unwrap_text_like_wrappers(s)
    s = _strip_numeric_trailing_units(s)
    s = _strip_terminal_ordinal_suffix(s)
    s = _strip_terminal_degree_markers(s)

    if r"\%" in s or "%" in s:
        return None

    i = 0
    out = []

    while i < len(s):
        if s[i] == "\\":
            j = i + 1
            while j < len(s) and (s[j].isalpha() or s[j] in ["\\"]):
                j += 1
            cmd = s[i:j]

            if cmd == r"\frac":
                while j < len(s) and s[j].isspace():
                    j += 1
                if j >= len(s) or s[j] != "{":
                    return None
                a, j2 = _extract_balanced_braces(s, j)
                if a is None:
                    return None
                while j2 < len(s) and s[j2].isspace():
                    j2 += 1
                if j2 >= len(s) or s[j2] != "{":
                    return None
                b, j3 = _extract_balanced_braces(s, j2)
                if b is None:
                    return None

                a_py = _latex_to_python_expr_general(a)
                b_py = _latex_to_python_expr_general(b)
                if a_py is None or b_py is None:
                    return None

                out.append(f"(({a_py})/({b_py}))")
                i = j3
                continue

            if cmd == r"\sqrt":
                while j < len(s) and s[j].isspace():
                    j += 1
                if j >= len(s) or s[j] != "{":
                    return None
                a, j2 = _extract_balanced_braces(s, j)
                if a is None:
                    return None
                a_py = _latex_to_python_expr_general(a)
                if a_py is None:
                    return None
                out.append(f"sqrt({a_py})")
                i = j2
                continue

            if cmd == r"\pi":
                out.append("pi")
                i = j
                continue

            if cmd in [r"\cdot", r"\times"]:
                out.append("*")
                i = j
                continue

            if cmd in [r"\div"]:
                out.append("/")
                i = j
                continue

            return None
        else:
            out.append(s[i])
            i += 1

    expr = "".join(out)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    expr = re.sub(r"\s+", "", expr)

    return expr or None


def _sympy_canonical_math(s: str) -> Optional[str]:
    if not _SYMPY_AVAILABLE:
        return None

    if "_" in s:
        return None

    expr = _latex_to_python_expr_general(s)
    if not expr:
        return None

    try:
        parsed = parse_expr(
            expr,
            local_dict={"sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E},
            transformations=_SYMPY_TRANSFORMS,
            evaluate=True,
        )
        simplified = sp.simplify(parsed)
        return sp.sstr(simplified)
    except Exception:
        return None


def _normalize_pmatrix(s: str) -> Optional[str]:
    m = re.fullmatch(
        r"\\begin\{pmatrix\}(.*)\\end\{pmatrix\}",
        s.strip(),
        flags=re.DOTALL,
    )
    if not m:
        return None

    body = m.group(1).strip()
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]

    canon_rows = []
    for row in rows:
        cells = [c.strip() for c in row.split("&")]
        canon_cells = []
        for cell in cells:
            canon_cells.append(_normalize_answer_core(cell, allow_matrix=False))
        canon_rows.append(",".join(canon_cells))

    return "pmatrix[" + ";".join(canon_rows) + "]"


def _normalize_answer_core(answer: str, allow_matrix: bool = True) -> str:
    if not answer:
        return ""

    s = answer.strip()

    if len(s) >= 2 and s[0] == "$" and s[-1] == "$" and s.count("$") >= 2:
        s = s[1:-1].strip()

    s = _strip_latex_noise(s)
    s = _normalize_frac_sqrt_and_subscript_syntax(s)
    s = _unwrap_text_like_wrappers(s)
    s = _strip_outer_braces_if_balanced(s)
    s = _strip_trailing_punct(s)
    s = _strip_terminal_ordinal_suffix(s)
    s = _strip_terminal_degree_markers(s)
    s = _strip_numeric_trailing_units(s)

    if allow_matrix:
        mat = _normalize_pmatrix(s)
        if mat is not None:
            return mat

    plain = _normalize_plain_text_answer(s)
    if plain is not None:
        return plain

    sym = _sympy_canonical_math(s)
    if sym is not None:
        return sym

    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = _LATEX_SPACE_CMDS_RE.sub("", s)
    s = re.sub(r"\s+", "", s)

    if _NUM_ONLY_RE.match(s):
        s_num = s.replace(",", "")
        try:
            d = Decimal(s_num)
            return format(d.normalize(), "f")
        except InvalidOperation:
            return s_num

    return s


def normalize_answer(answer) -> str:
    """Normalize an answer for comparison. Non-string inputs (e.g. AIME integer
    ground truths) are coerced to str; None becomes ''."""
    answer = "" if answer is None else str(answer)
    return _normalize_answer_core(answer, allow_matrix=True)


def _strip_latex_noise(s: str) -> str:
    s = s.strip()

    if len(s) >= 2 and s[0] == "$" and s[-1] == "$" and s.count("$") >= 2:
        s = s[1:-1].strip()

    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "").replace(r"\!", "")
    s = re.sub(r"\\(?:quad|qquad)\b", "", s)
    s = re.sub(r"\\(?:displaystyle|textstyle)\b", "", s)

    s = _unwrap_text_like_wrappers(s)
    s = re.sub(r"\\([$%#&_{}])", r"\1", s)

    s = (
        s.replace("−", "-")
         .replace("–", "-")
         .replace("×", "*")
         .replace("·", "*")
         .replace("÷", "/")
    )

    return s


def _insert_implicit_mul(expr: str) -> str:
    e = expr
    e = re.sub(r"(\d|\))\s*(\()", r"\1*\2", e)
    e = re.sub(r"(\))\s*(\d)", r"\1*\2", e)
    e = re.sub(r"(\d|\))\s*(pi|e|sqrt)\b", r"\1*\2", e)
    e = re.sub(r"\b(pi|e)\s*(\()", r"\1*\2", e)
    return e


def latex_to_python_numeric_expr(s: str) -> Optional[str]:
    if not s:
        return None

    s = _strip_latex_noise(s)

    if any(tok in s for tok in ["=", "<", ">", "\\pm", "±", "\\in", "\\cup", "\\cap"]):
        return None
    if "{" in s and "}" in s and "," in s:
        return None

    i = 0
    out = []
    while i < len(s):
        if s[i] == "\\":
            j = i + 1
            while j < len(s) and (s[j].isalpha() or s[j] in ["\\"]):
                j += 1
            cmd = s[i:j]

            if cmd in [r"\frac", r"\dfrac", r"\tfrac"]:
                while j < len(s) and s[j].isspace():
                    j += 1
                if j >= len(s) or s[j] != "{":
                    return None
                a, j2 = _extract_balanced_braces(s, j)
                if a is None:
                    return None
                while j2 < len(s) and s[j2].isspace():
                    j2 += 1
                if j2 >= len(s) or s[j2] != "{":
                    return None
                b, j3 = _extract_balanced_braces(s, j2)
                if b is None:
                    return None

                a_py = latex_to_python_numeric_expr(a)
                b_py = latex_to_python_numeric_expr(b)
                if a_py is None or b_py is None:
                    return None
                out.append(f"(({a_py})/({b_py}))")
                i = j3
                continue

            if cmd == r"\sqrt":
                k = j
                while k < len(s) and s[k].isspace():
                    k += 1

                n_root = None
                if k < len(s) and s[k] == "[":
                    k2 = s.find("]", k + 1)
                    if k2 == -1:
                        return None
                    n_root = s[k + 1 : k2].strip()
                    k = k2 + 1
                    while k < len(s) and s[k].isspace():
                        k += 1

                if k >= len(s) or s[k] != "{":
                    return None
                a, k2 = _extract_balanced_braces(s, k)
                if a is None:
                    return None
                a_py = latex_to_python_numeric_expr(a)
                if a_py is None:
                    return None

                if n_root is None:
                    out.append(f"sqrt({a_py})")
                else:
                    n_py = latex_to_python_numeric_expr(n_root)
                    if n_py is None:
                        return None
                    out.append(f"(({a_py})**(1/({n_py})))")
                i = k2
                continue

            if cmd == r"\pi":
                out.append("pi")
                i = j
                continue

            if cmd in [r"\cdot", r"\times"]:
                out.append("*")
                i = j
                continue

            if cmd in [r"\div"]:
                out.append("/")
                i = j
                continue

            if len(cmd) > 1 and cmd[1:].isalpha():
                return None
            i = j
            continue

        else:
            out.append(s[i])
            i += 1

    expr = "".join(out)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    expr = re.sub(r"\s+", "", expr)
    expr = _insert_implicit_mul(expr)

    tmp = re.sub(r"\b(sqrt|pi|e)\b", "", expr)
    if re.search(r"[A-Za-z]", tmp):
        return None
    if re.search(r"[^\d+\-*/().**]", tmp):
        return None
    if "****" in expr:
        return None

    return expr


def safe_eval_numeric(expr: str) -> Optional[float]:
    if not expr:
        return None

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    allowed_names = {"pi": math.pi, "e": math.e}
    allowed_funcs = {"sqrt": math.sqrt}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise TypeError("Non-numeric constant")

        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left = _eval(node.left)
            right = _eval(node.right)
            return ops[type(node.op)](left, right)

        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return float(allowed_names[node.id])
            raise TypeError("Unknown name")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("Unsafe call")
            fn = node.func.id
            if fn not in allowed_funcs:
                raise TypeError("Unknown function")
            if len(node.args) != 1:
                raise TypeError("Wrong arity")
            return float(allowed_funcs[fn](_eval(node.args[0])))

        raise TypeError(f"Unsupported AST node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        return float(_eval(tree))
    except Exception:
        return None


def _strip_terminal_percent_or_degree(s: str) -> tuple[str, Optional[str]]:
    r"""
    Returns (base_expr, kind) where kind is one of:
      - "percent"
      - "degree"
      - None
    Examples:
      "60\\%"     -> ("60", "percent")
      "82^\\circ" -> ("82", "degree")
      "82^{\\circ}" -> ("82", "degree")
      "82°"       -> ("82", "degree")
      "60"        -> ("60", None)
    """
    if not s:
        return s, None

    x = s.strip()

    # percent: 60\% or 60%
    y = re.sub(r"(?:\\%|%)\s*$", "", x)
    if y != x:
        return y.strip(), "percent"

    # degree: 82^\circ / 82^{\circ} / 82°
    y = re.sub(r"(?:\^\{\\circ\}|\^\\circ|°)\s*$", "", x)
    if y != x:
        return y.strip(), "degree"

    return x, None


def _try_same_value_with_optional_percent_or_degree(a: str, b: str) -> bool:
    r"""
    Treat these as equivalent:
      60 == 60\%
      82 == 82^\circ

    But only when the underlying numeric values are equal.
    This is separate from the usual percent semantics:
      25\% == 0.25
    """
    if not a or not b:
        return False

    a_base, a_kind = _strip_terminal_percent_or_degree(a)
    b_base, b_kind = _strip_terminal_percent_or_degree(b)

    # Need exactly one side to have the suffix, or both same kind.
    # This keeps the rule flexible and symmetric.
    if a_kind is None and b_kind is None:
        return False

    a_expr = latex_to_python_numeric_expr(a_base)
    b_expr = latex_to_python_numeric_expr(b_base)
    if not a_expr or not b_expr:
        return False

    av = safe_eval_numeric(a_expr)
    bv = safe_eval_numeric(b_expr)
    if av is None or bv is None:
        return False

    diff = abs(av - bv)
    tol = 1e-9 + 1e-9 * max(1.0, abs(bv))
    return diff <= tol


def is_correct(predicted, ground_truth) -> bool:
    """Exact-match after normalization, with numeric / symbolic / percent-degree
    equivalences. Non-string inputs are coerced to str (None -> '' -> False)."""
    predicted = "" if predicted is None else str(predicted)
    ground_truth = "" if ground_truth is None else str(ground_truth)
    if not predicted or not ground_truth:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # 60 == 60\%
    # 82 == 82^\circ
    if _try_same_value_with_optional_percent_or_degree(pred_norm, gt_norm):
        return True

    # 25\% == 0.25
    def _try_percent_equiv(a: str, b: str) -> bool:
        if a.endswith(r"\%") and not b.endswith(r"\%"):
            a_base = a[:-2].strip()
            a_sym = _sympy_canonical_math(a_base)
            b_sym = _sympy_canonical_math(b)
            if a_sym is not None and b_sym is not None and _SYMPY_AVAILABLE:
                try:
                    av = sp.nsimplify(a_sym)
                    bv = sp.nsimplify(b_sym)
                    return sp.simplify(av / 100 - bv) == 0
                except Exception:
                    pass

        if b.endswith(r"\%") and not a.endswith(r"\%"):
            a_sym = _sympy_canonical_math(a)
            b_base = b[:-2].strip()
            b_sym = _sympy_canonical_math(b_base)
            if a_sym is not None and b_sym is not None and _SYMPY_AVAILABLE:
                try:
                    av = sp.nsimplify(a_sym)
                    bv = sp.nsimplify(b_sym)
                    return sp.simplify(av - bv / 100) == 0
                except Exception:
                    pass

        return False

    if _try_percent_equiv(pred_norm, gt_norm):
        return True

    if _SYMPY_AVAILABLE:
        a_sym = _sympy_canonical_math(pred_norm)
        b_sym = _sympy_canonical_math(gt_norm)
        if a_sym is not None and b_sym is not None:
            try:
                return sp.simplify(sp.sympify(a_sym) - sp.sympify(b_sym)) == 0
            except Exception:
                pass

    if pred_norm.replace("**", "^") == gt_norm.replace("**", "^"):
        return True
    if pred_norm.replace("^", "**") == gt_norm.replace("^", "**"):
        return True

    return False