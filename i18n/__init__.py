"""
i18n layer — Manga-Translator (spec docs/i18n-v1-spec.md §3.3).

Server-side locale resolution + message-key dictionaries used by app.py
(progress events, errors, warnings) and by Jinja templates through the
context processor registered via register_app().

Fallback chain: locale dict -> vi dict -> raw key. t() is safe to call
before i18n/*.json exist (parallel t2/t3 development): missing files simply
resolve to the raw key, so the app never crashes on i18n setup.
"""
import json
import os
import re

from flask import g, request
from markupsafe import Markup

SUPPORTED_LOCALES = ["vi", "en"]
DEFAULT_LOCALE = "vi"

_I18N_DIR = os.path.dirname(os.path.abspath(__file__))
_ACCEPT_LANG_RE = re.compile(r"^\s*([a-zA-Z]{2,3})")
_VIETNAMESE_RE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]"
)

# Successful loads are cached forever; misses are re-read on every call so
# the files may appear later (development order t2/t3) without a restart.
_dict_cache = {}

class WarningMessage(dict):
    """Structured warning: {"key": "backend.warn.*", "params": {...}}.

    Stays a mapping (locale-aware templates/JS re-render via t(key, **params))
    while stringifying to the localized message, so templates that still use
    {{ warning }} degrade gracefully to the message of the current locale.
    """

    def __init__(self, key, params=None):
        super().__init__(key=key, params=params or {})

    def __str__(self):
        return translate(self.get("key") or "", get_locale(), **(self.get("params") or {}))

    __repr__ = __str__


def load_dict(locale):
    """Return the message dictionary for `locale` (cached). {} when missing."""
    data = _dict_cache.get(locale)
    if data is not None:
        return data
    path = os.path.join(_I18N_DIR, f"{locale}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        data = raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        data = {}
    if data:
        _dict_cache[locale] = data
    return data


def resolve_locale(cookie_val=None, accept_lang=""):
    """Pick the locale: valid cookie wins, else the first Accept-Language tag
    ("vi*" -> vi, "en*" -> en), else DEFAULT_LOCALE."""
    if cookie_val in SUPPORTED_LOCALES:
        return cookie_val
    match = _ACCEPT_LANG_RE.match(str(accept_lang or ""))
    if match:
        prefix = match.group(1).lower()
        if prefix == "vi":
            return "vi"
        if prefix == "en":
            return "en"
    return DEFAULT_LOCALE


def get_locale():
    """Locale of the current request (g.locale), or DEFAULT_LOCALE outside a
    request context (tests, background threads)."""
    try:
        locale = getattr(g, "locale", None)
    except Exception:
        locale = None
    return locale if locale in SUPPORTED_LOCALES else DEFAULT_LOCALE


def translate(key, locale=None, **params):
    """Resolve `key` for `locale` with the chain locale -> vi -> raw key.

    {name} placeholders are filled from params. Keys ending in _html
    return Markup (safe for Jinja); all other keys are plain text.
    """
    locale = locale or get_locale()
    value = None
    for candidate in (load_dict(locale), load_dict(DEFAULT_LOCALE)):
        found = candidate.get(key)
        if found is not None:
            value = found
            break
    if value is None:
        value = key
    if params:
        try:
            value = str(value).format(**params)
        except (KeyError, IndexError, ValueError):
            pass  # unresolved placeholders: keep the raw template
    if key.endswith("_html"):
        return Markup(value)
    return value


def t(key, **params):
    """Localize `key` for the current request locale (context processor helper)."""
    return translate(key, get_locale(), **params)


def tp(key_base, n, **params):
    """Plural-aware localization: <key_base>_one/_other chosen by the locale's
    plural rule (en: n == 1 -> one; vi always other). Missing _one falls back
    to _other."""
    locale = get_locale()
    plural = "one" if (locale == "en" and n == 1) else "other"
    full = f"{key_base}_{plural}"
    # The count is always injected as the {n} placeholder; an explicit n in
    # params (matching the JS I18N.tp contract) is honored unchanged.
    params.setdefault("n", n)
    value = translate(full, locale, **params)
    if value == full and plural == "one":
        value = translate(f"{key_base}_other", locale, **params)
    return value


def i18n_json():
    """Both dictionaries as JSON for inline embedding (spec §3.4).

    "</" is escaped so the payload is safe inside a <script> tag.
    """
    payload = {locale: load_dict(locale) for locale in SUPPORTED_LOCALES}
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def register_app(flask_app):
    """Wire the i18n layer into a Flask app: before_request locale resolution,
    teardown state cleanup, and the context processor injecting t/tp/
    current_locale/i18n_json into every template."""

    @flask_app.before_request
    def _i18n_set_locale():
        g.locale = resolve_locale(
            request.cookies.get("mt_locale"),
            request.headers.get("Accept-Language", ""),
        )

    @flask_app.teardown_request
    def _i18n_clear_locale(exc=None):
        try:
            g.pop("locale", None)
        except Exception:
            pass

    @flask_app.context_processor
    def _i18n_context_processor():
        return {
            "t": t,
            "tp": tp,
            "current_locale": get_locale(),
            "i18n_json": i18n_json(),
        }
