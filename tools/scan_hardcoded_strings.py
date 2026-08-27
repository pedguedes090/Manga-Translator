#!/usr/bin/env python3
"""scan_hardcoded_strings.py — find leftover hardcoded user-visible strings.

Spec: docs/i18n-v1-spec.md (A0.2)
Scans the frontend/backend sources that should render ONLY through the i18n
dictionaries:
  - templates/*.html
  - static/js/*.js
  - app.py
  - translator/gemini_translator.py

Detects:
  1. Vietnamese text (diacritics) outside comments and outside the allowlist.
  2. Known English UI tokens inside string literals / text nodes that now have
     dictionary keys (e.g. "Translate", "Download", "Cancel").

Excluded by design: i18n/, static/qa/, tests/, temp_sessions/, docs/,
debug_outputs/, comment lines, pure-number/emoji/code strings, proper nouns
(font names, language names, model names, brand names).

Usage: python tools/scan_hardcoded_strings.py [--json]
Exit code: 0 = clean, 1 = findings.
"""
import argparse
import json
import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

HTML_FILES = sorted((ROOT / "templates").glob("*.html"))
JS_FILES = sorted((ROOT / "static" / "js").glob("*.js"))
PY_FILES = [ROOT / "app.py", ROOT / "translator" / "gemini_translator.py"]
ALL_FILES = HTML_FILES + JS_FILES + [p for p in PY_FILES if p.exists()]

VIET = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]"
)
# English UI tokens that must come from the dictionaries now. Deliberately
# avoids generic words that also appear in code/identifiers ("blocks", "OCR",
# "models", "images").
UI_TOKENS = [
    "Translate", "Download ZIP", "Download complete", "Creating ZIP",
    "ZIP creation failed", "Cancel", "Undo", "Redo", "Reset", "Retry",
    "Preparing", "Rendering image", "Rendering text", "Session expired",
    "Speech bubble", "speech bubbles", "Choose images", "images selected",
    "more images", "Custom prompt", "Model name", "Server URL", "API key",
    "Source language", "Translate to", "Translation style", "Manual correction",
    "Processing... Please wait", "OCR running", "OCR error", "No text recognized",
    "not rendered", "not saved", "Back to results", "Back to OCR editing",
    "Continue translating", "Properties", "Text style", "Font size",
    "Text color", "Bold/italic", "Alignment", "Apply to all blocks",
    "Invalid bbox", "Could not re-render", "Edit speech bubbles",
    "Post-translation edit", "Edit & Style", "Zoom controls", "Fit to screen",
    "Actual size", "Go to image", "Compare image", "Translation results",
    "No images were processed", "Page title",
]

# Allowlisted full substrings: proper nouns / values that stay untranslated.
ALLOW_SUBSTRINGS = [
    "http://localhost", "Ollama", "LM Studio", "LocalAI", "Gemini", "Local LLM",
    "Google Translate", "Animeace", "Mangat", "Arial", "Yuki-", "Japanese (Manga)",
    "Chinese (Manhua)", "Korean (Manhwa)", "English (Comic)", "Vietnamese",
    "English", "Chinese", "Korean", "Thai", "Indonesian", "French", "German",
    "Spanish", "Russian", "llama3.2", "qwen2.5", "mistral", "gemini-3.1",
    "gpt-4", "manga_translated", "_translated", "Manga Translator",
    "manga_translated.zip", "Manga-Translator", "Tiếng Việt",
]

# Legacy form values kept as ADDITIVE alias map keys (spec 5.6: "giữ key cũ")
# so old submissions (display text) keep working after the data-value refactor.
# These are DATA lookups, never rendered UI.
LEGACY_FORM_VALUES = {
    "japanese (manga)", "chinese (manhua)", "korean (manhwa)", "english (comic)",
    "vietnamese", "english", "chinese", "korean", "thai", "indonesian",
    "french", "german", "spanish", "russian",
    "local llm", "copilot", "gemini", "google",
    "default", "casual (thân mật)", "formal (trang trọng)",
    "keep honorifics (-san, senpai...)", "web novel style",
    "action (ngắn gọn)", "literal (sát nghĩa)", "custom...",
    "auto (match original)", "animeace",
}

# Out of scope per spec section 9: LLM translation prompts (gemini_translator
# prompt block) and print() console logs — never rendered to the UI.
OUT_OF_SCOPE_STRINGS = [
    "Retrying in {delay}s...",
    '["bản dịch 1", "bản dịch 2", ...]',
    # Prompt format example (spec 9: LLM prompts are out of scope).
    "bản dịch 1",
    "bản dịch 2",
]


def is_clean(s):
    """True when the string needs no dictionary key (proper noun / legacy data /
    out-of-scope content / pure numbers)."""
    if not s or len(s) < 3:
        return True
    if s.strip() in LEGACY_FORM_VALUES:
        return True
    if any(o in s for o in OUT_OF_SCOPE_STRINGS):
        return True
    rest = s
    for a in sorted(ALLOW_SUBSTRINGS, key=len, reverse=True):
        rest = rest.replace(a, " ")
    return not re.search(r"\w", rest)


def strip_html_comments(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def strip_js_py_comments(text):
    # line comments (//, #) and block comments (/* */)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        # strip trailing // comments (keep URLs like http://)
        if "//" in line and not re.search(r"https?://", line):
            line = re.split(r"//", line)[0]
        lines.append(line)
    return "\n".join(lines)


def html_text_nodes(content):
    # text between tags
    return re.findall(r">([^<>]{1,200})<", content)


def html_attr_values(content):
    attrs = re.findall(
        r'(?:title|aria-label|placeholder|alt|label|data-original-label)\s*=\s*"([^"]*)"',
        content,
    )
    return attrs


def quoted_strings(text):
    return re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"", text)


def is_ui_hit(s):
    if is_clean(s):
        return False
    if re.fullmatch(r"[\d\s%.,:+×x()/\\-]+", s):
        return False
    if VIET.search(s):
        return True
    for tok in UI_TOKENS:
        if tok in s:
            return True
    return False


def scan_file(path):
    findings = []
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        content = strip_html_comments(content)
        candidates = html_text_nodes(content) + html_attr_values(content)
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            if "{{" in cand or "{%" in cand or "<" in cand or ">" in cand:
                continue
            if is_clean(cand):
                continue
            if is_ui_hit(cand):
                findings.append((cand, "text/attr"))
    else:
        content = strip_js_py_comments(content)
        for line_no, line in enumerate(content.splitlines(), 1):
            for q in quoted_strings(line):
                s = (q[0] or q[1] or "").strip()
                if not s or len(s) < 3:
                    continue
                if re.fullmatch(r"[\d\s%.,:+×x()/\\-]+", s):
                    continue
                if is_clean(s):
                    continue
                if is_ui_hit(s):
                    findings.append((s, f"line {line_no}"))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = {}
    total = 0
    for path in ALL_FILES:
        findings = scan_file(path)
        if findings:
            results[str(path.relative_to(ROOT))] = findings
            total += len(findings)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("OK: no hardcoded user-visible strings found")
        else:
            print(f"FAIL: {total} finding(s):")
            for path, items in results.items():
                print(f"  {path}:")
                for s, where in items:
                    print(f"    - {where}: {s[:110]}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
