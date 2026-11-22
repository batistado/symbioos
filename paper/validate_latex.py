#!/usr/bin/env python3
"""Static LaTeX, BibTeX, IEEE TPDS Desk-Rejection & Humanizer Prose Validator for SymbioOS."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import List, Set, Tuple

PAPER_DIR = Path(__file__).resolve().parent

BANNED_AI_WORDS = (
    "crucial",
    "pivotal",
    "delve",
    "delves",
    "delving",
    "landscape",
    "testament",
    "showcase",
    "showcases",
    "showcasing",
    "underscore",
    "underscores",
    "underscoring",
    "interplay",
    "intricate",
    "intricacies",
    "meticulous",
    "meticulously",
    "vibrant",
    "tapestry",
    "garner",
    "bolstered",
)


def strip_comments(tex: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for idx, raw in enumerate(tex.splitlines(), start=1):
        out = []
        i = 0
        while i < len(raw):
            if raw[i] == "\\" and i + 1 < len(raw):
                out.append(raw[i : i + 2])
                i += 2
            elif raw[i] == "%":
                break
            else:
                out.append(raw[i])
                i += 1
        lines.append((idx, "".join(out)))
    return lines


def validate_tex_and_bib(tex_path: Path, bib_path: Path) -> int:
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    clean_lines = strip_comments(tex)
    clean_tex = "\n".join(line for _, line in clean_lines)

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Brace balance
    depth = 0
    stack_lines: List[int] = []
    for lno, line in clean_lines:
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                i += 2
                continue
            if line[i] == "{":
                depth += 1
                stack_lines.append(lno)
            elif line[i] == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"Unmatched closing '}}' at line {lno}")
                    depth = 0
                else:
                    stack_lines.pop()
            i += 1
    if depth != 0:
        errors.append(f"Unclosed '{{' braces remaining: {depth} (opened at lines {stack_lines[-5:]})")

    # 2. Environment nesting
    env_stack: List[Tuple[str, int]] = []
    env_re = re.compile(r"\\(begin|end)\{([^}]+)\}")
    for lno, line in clean_lines:
        for m in env_re.finditer(line):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                env_stack.append((name, lno))
            else:
                if not env_stack:
                    errors.append(f"Unmatched \\end{{{name}}} at line {lno}")
                else:
                    top_name, top_line = env_stack.pop()
                    if top_name != name:
                        errors.append(
                            f"Environment mismatch: \\begin{{{top_name}}} (line {top_line}) "
                            f"closed by \\end{{{name}}} (line {lno})"
                        )
    for name, lno in env_stack:
        errors.append(f"Unclosed \\begin{{{name}}} opened at line {lno}")

    # 3. Inline '$' math parity
    dollar_count = 0
    i = 0
    while i < len(clean_tex):
        if clean_tex[i] == "\\" and i + 1 < len(clean_tex):
            i += 2
            continue
        if clean_tex[i] == "$":
            if i + 1 < len(clean_tex) and clean_tex[i + 1] == "$":
                errors.append("Found TeX display '$$' instead of LaTeX equation/align")
                i += 2
                continue
            dollar_count += 1
        i += 1
    if dollar_count % 2 != 0:
        errors.append(f"Odd number of unescaped '$' math delimiters ({dollar_count})")

    # 4. Markdown backticks, unescaped '_' / '#', & Humanizer prose checks
    in_disp_math = False
    id_cmd_re = re.compile(
        r"\\(?:label|ref|eqref|cite\w*|includegraphics(?:\[[^\]]*\])?|url|bibliography|bibliographystyle|graphicspath|input)\{[^}]*\}"
    )
    for lno, line in clean_lines:
        if re.search(r"\\begin\{(?:equation\*?|align\*?|multline\*?)\}", line):
            in_disp_math = True
        if not in_disp_math and not re.match(r"^\s*\\(?:newcommand|renewcommand|providecommand|def)\b", line):
            tmp = line.replace("``", "")
            if "`" in tmp:
                errors.append(f"Markdown single backtick found at line {lno}: {line.strip()[:80]}")
            if "---" in line:
                errors.append(f"Humanizer check: em-dash '---' found at line {lno}")
            stripped = id_cmd_re.sub("", line)
            stripped = re.sub(r"\$[^$]*\$", "", stripped)
            j = 0
            while j < len(stripped):
                if stripped[j] == "\\" and j + 1 < len(stripped):
                    j += 2
                    continue
                if stripped[j] in ("_", "#"):
                    errors.append(
                        f"Unescaped '{stripped[j]}' outside math mode at line {lno}: {line.strip()[:80]}"
                    )
                    break
                j += 1
            low = stripped.lower()
            for bw in BANNED_AI_WORDS:
                if re.search(rf"\b{re.escape(bw)}\b", low):
                    errors.append(f"Humanizer check: banned stock AI word '{bw}' at line {lno}")
            if re.search(r"\bnot\s+(?:just|merely|only)\b", low):
                errors.append(f"Humanizer check: 'not just/merely/only' formula at line {lno}")
        if re.search(r"\\end\{(?:equation\*?|align\*?|multline\*?)\}", line):
            in_disp_math = False

    # 5. Strict IEEE TPDS Abstract Gate
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", clean_tex, flags=re.S)
    abs_words = 0
    if not abs_match:
        errors.append("Missing \\begin{abstract}...\\end{abstract}")
    else:
        abs_raw = abs_match.group(1).strip()
        if "\n\n" in abs_raw:
            errors.append("IEEE TPDS abstract must be a single paragraph (found blank line)")
        if "$" in abs_raw or "\\begin{equation" in abs_raw:
            errors.append("IEEE TPDS abstract must contain zero mathematical formulas / '$'")
        if "\\cite" in abs_raw or "\\footnote" in abs_raw:
            errors.append("IEEE TPDS abstract must contain zero citations or footnotes")
        abs_plain = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", abs_raw)
        abs_plain = re.sub(r"[{}~]", " ", abs_plain)
        abs_words = len(abs_plain.split())
        if abs_words > 200:
            errors.append(f"IEEE TPDS abstract exceeds 200 words ({abs_words} words)")
        elif abs_words < 140:
            warnings.append(f"IEEE TPDS abstract is short ({abs_words} words; target 155-195)")

    # 6. IEEE TPDS Mandatory Metadata & Disclosure Gates
    if "0009-0009-9058-5375" not in clean_tex:
        errors.append("Missing corresponding author ORCID in IEEEcompsocitemizethanks")
    if r"\begin{IEEEkeywords}" not in clean_tex:
        errors.append("Missing \\begin{IEEEkeywords} block")
    if "Availability" not in clean_tex:
        errors.append("Missing Data and Code Availability statement")
    if "Competing Interests" not in clean_tex and "Acknowledgments" not in clean_tex:
        errors.append("Missing Acknowledgments / Competing Interests disclosure")
    bio_count = len(re.findall(r"\\begin\{IEEEbiographynophoto\}", clean_tex))
    if bio_count != 5:
        errors.append(f"Expected 5 IEEEbiographynophoto entries, found {bio_count}")

    # 7. Citations vs BibTeX keys
    bib_keys: Set[str] = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))
    cited_keys: Set[str] = set()
    for m in re.finditer(r"\\cite\w*\{([^}]+)\}", clean_tex):
        for k in m.group(1).split(","):
            k_clean = k.strip()
            if k_clean:
                cited_keys.add(k_clean)
                if k_clean not in bib_keys:
                    errors.append(f"Undefined citation key: '{k_clean}'")
    unused_bib = sorted(bib_keys - cited_keys)
    if unused_bib:
        warnings.append(f"Unused BibTeX entries ({len(unused_bib)}): {', '.join(unused_bib)}")

    bib_depth = 0
    for ch in bib:
        if ch == "{":
            bib_depth += 1
        elif ch == "}":
            bib_depth -= 1
    if bib_depth != 0:
        errors.append(f"BibTeX brace imbalance in {bib_path.name}: {bib_depth}")

    # 8. Labels vs Refs
    labels: Set[str] = set()
    for m in re.finditer(r"\\label\{([^}]+)\}", clean_tex):
        lbl = m.group(1).strip()
        if lbl in labels:
            errors.append(f"Duplicate \\label{{{lbl}}}")
        labels.add(lbl)
    referenced: Set[str] = set()
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", clean_tex):
        ref = m.group(1).strip()
        referenced.add(ref)
        if ref not in labels:
            errors.append(f"Undefined reference target: \\ref{{{ref}}}")
    unused_labels = sorted(labels - referenced)
    if unused_labels:
        warnings.append(f"Unreferenced labels ({len(unused_labels)}): {', '.join(unused_labels)}")

    # 9. Graphics files
    gfx_files: List[str] = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex):
        rel = m.group(1).strip()
        gfx_files.append(rel)
        cand = PAPER_DIR / "figures" / rel
        if not cand.exists() and not (PAPER_DIR / rel).exists():
            errors.append(f"Missing figure file: {rel}")

    # 10. Calibrated IEEE TPDS (`10pt,journal,compsoc`) Page Model
    body_part = (
        clean_tex.split(r"\maketitle")[1].split(r"\bibliographystyle")[0]
        if r"\maketitle" in clean_tex and r"\bibliographystyle" in clean_tex
        else clean_tex
    )
    prose_body = body_part
    for env in ("table*", "table", "figure*", "figure", "equation", "align"):
        pat = re.escape(f"\\begin{{{env}}}") + r".*?" + re.escape(f"\\end{{{env}}}")
        prose_body = re.sub(pat, " ", prose_body, flags=re.S)
    prose_body = re.sub(r"\\(?:label|ref|eqref)\{[^}]*\}", " 1 ", prose_body)
    prose_body = re.sub(r"\\cite\w*\{[^}]*\}", " [1] ", prose_body)
    prose_body = re.sub(r"\$[^$]+\$", " MATH_EQ ", prose_body)
    prose_body = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", prose_body)
    prose_body = re.sub(r"[{}\\~]", " ", prose_body)
    rendered_prose_words = len(prose_body.split())

    num_fig_star = len(re.findall(r"\\begin\{figure\*\}", body_part))
    num_fig_col = len(re.findall(r"\\begin\{figure\}", body_part))
    num_tab_star = len(re.findall(r"\\begin\{table\*\}", body_part))
    num_tab_col = len(re.findall(r"\\begin\{table\}", body_part))
    num_disp_eq = len(re.findall(r"\\begin\{(?:equation|align)\}", body_part))
    num_thms = len(re.findall(r"\\begin\{(?:theorem|lemma|proposition|corollary|remark|IEEEproof)\}", body_part))
    num_sec = len(re.findall(r"\\(?:sub)*section\{", body_part))

    est_pages = (
        0.52
        + (rendered_prose_words / 1150.0)
        + (num_fig_star * 0.30)
        + (num_fig_col * 0.20)
        + (num_tab_star * 0.32)
        + (num_tab_col * 0.16)
        + (num_disp_eq * 0.03)
        + (num_thms * 0.025)
        + (num_sec * 0.015)
        + (len(cited_keys) * 0.022)
        + (bio_count * 0.06)
    )

    print("=== SymbioOS IEEE TPDS LaTeX & Humanizer Validation Report ===")
    print(f"  LaTeX source lines       : {len(tex.splitlines())}")
    print(f"  Abstract word count      : {abs_words} words (IEEE TPDS limit <= 200)")
    print(f"  Rendered body word equiv : {rendered_prose_words}")
    print(f"  Double-col figures       : {num_fig_star}, Single-col figures: {num_fig_col}")
    print(f"  Double-col tables        : {num_tab_star}, Single-col tables : {num_tab_col}")
    print(f"  Displayed eqs / thms     : {num_disp_eq} eqs, {num_thms} theorem/proof blocks")
    print(f"  Citations / Bib keys     : {len(cited_keys)} cited / {len(bib_keys)} defined")
    print(f"  Labels / Cross-refs      : {len(labels)} labels / {len(referenced)} referenced")
    print(f"  Included PDF figures     : {len(gfx_files)} ({', '.join(gfx_files)})")
    print(f"  Author biographies       : {bio_count}")
    print(f"  Estimated TPDS pages     : ~{est_pages:.2f} pages (10pt,journal,compsoc)")

    for w in warnings:
        print(f"  [WARN] {w}")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        return 1
    print("  [PASS] Zero brace, environment, abstract, metadata, citation, label, figure, or humanizer errors!")
    return 0


if __name__ == "__main__":
    sys.exit(validate_tex_and_bib(PAPER_DIR / "main.tex", PAPER_DIR / "references.bib"))
