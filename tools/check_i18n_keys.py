#!/usr/bin/env python3
"""check_i18n_keys.py — verify locale dictionary integrity for Manga-Translator.

Spec: docs/i18n-v1-spec.md (A0.1, A1.5)
Checks:
  1. Every locale file in i18n/*.json exists and is valid JSON.
  2. Key sets are IDENTICAL across locales (key parity).
  3. Placeholder parity: every {x} placeholder in one locale's value must exist
     in the same key of every other locale (and vice versa).
  4. Plural pairs: every *_one key has a *_other sibling and vice versa.
  5. Values are strings (no nested structures).

Exit code 0 = OK, 1 = violations found.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I18N_DIR = ROOT / "i18n"
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

def main():
    json_files = sorted(I18N_DIR.glob("*.json"))
    if not json_files:
        print(f"FAIL: no locale JSON files found in {I18N_DIR}")
        sys.exit(1)

    dicts = {}
    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"FAIL: {p.name} is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(data, dict):
            print(f"FAIL: {p.name} must be a JSON object")
            sys.exit(1)
        for k, v in data.items():
            if not isinstance(v, str):
                print(f"FAIL: {p.name}: key {k!r} value must be a string")
                sys.exit(1)
        dicts[p.stem] = data

    errors = []
    base_locale = sorted(dicts)[0]
    base = dicts[base_locale]

    # 1. Key parity
    for loc, data in dicts.items():
        if loc == base_locale:
            continue
        missing = sorted(set(base) - set(data))
        extra = sorted(set(data) - set(base))
        if missing:
            errors.append(f"{loc}: missing keys: {', '.join(missing)}")
        if extra:
            errors.append(f"{loc}: extra keys: {', '.join(extra)}")

    # 2. Placeholder parity
    for key in sorted(base):
        base_ph = sorted(set(PLACEHOLDER_RE.findall(base[key])))
        for loc, data in dicts.items():
            if key not in data:
                continue
            loc_ph = sorted(set(PLACEHOLDER_RE.findall(data[key])))
            if base_ph != loc_ph:
                errors.append(
                    f"{loc}: placeholder mismatch for {key!r}: "
                    f"{base_locale}={base_ph} {loc}={loc_ph}"
                )

    # 3. Plural pairs (_one/_other) in every locale
    for loc, data in dicts.items():
        keys = set(data)
        for k in sorted(keys):
            if k.endswith("_one") and k[:-4] + "_other" not in keys:
                errors.append(f"{loc}: {k!r} has no _other pair")
            if k.endswith("_other") and k[:-6] + "_one" not in keys:
                errors.append(f"{loc}: {k!r} has no _one pair")

    if errors:
        print(f"FAIL: {len(errors)} issue(s) found")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"OK: {len(base)} keys, key parity across {len(dicts)} locale(s), "
          f"placeholder parity OK, plural pairs OK")
    sys.exit(0)

if __name__ == "__main__":
    main()
