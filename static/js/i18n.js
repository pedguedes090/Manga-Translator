/*!
 * i18n.js — Manga-Translator frontend i18n runtime (vi/en).
 * Spec: docs/i18n-v1-spec.md §3.4/§3.6.
 * - Reads the embedded dictionaries from <script id="i18n-data" type="application/json">.
 * - t()/tp() with locale -> vi -> raw-key fallback (never throws).
 * - Auto-detect navigator.language on first visit, persisted via localStorage
 *   (mt_locale) + cookie (mt_locale); user choice wins over detection.
 * - Injects the language dropdown into <header> (index/translate) or
 *   .corr-topbar (correction); change = persist + reload (guarded, max 1).
 * - onRefresh()/i18n:changed hooks for live re-render of JS-driven strings (P1).
 */
(function () {
    'use strict';

    var SUPPORTED = ['vi', 'en'];
    var DEFAULT_LOCALE = 'vi';
    var STORAGE_KEY = 'mt_locale';
    var COOKIE_KEY = 'mt_locale';
    var RELOAD_GUARD = 'mt_i18n_reloaded';
    // Native names for the dropdown (new locale = +1 entry here + 1 JSON file).
    // F3 (UI gate): globe prefix makes the language switcher discoverable.
    var LOCALE_LABELS = { vi: '🌐 Tiếng Việt', en: '🌐 English' };

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function parseDicts() {
        var dicts = { vi: {}, en: {} };
        var el = document.getElementById('i18n-data');
        if (el && el.textContent) {
            try {
                var data = JSON.parse(el.textContent);
                if (data && typeof data === 'object') {
                    SUPPORTED.forEach(function (loc) {
                        if (data[loc] && typeof data[loc] === 'object') dicts[loc] = data[loc];
                    });
                }
            } catch (e) {
                /* empty dicts -> t() returns raw keys; page still works */
            }
        }
        return dicts;
    }

    function lookup(locale, key) {
        var dict = I18N.dicts[locale];
        if (dict && Object.prototype.hasOwnProperty.call(dict, key)) return dict[key];
        if (locale !== DEFAULT_LOCALE) {
            var viDict = I18N.dicts[DEFAULT_LOCALE];
            if (viDict && Object.prototype.hasOwnProperty.call(viDict, key)) return viDict[key];
        }
        return null;
    }

    function pluralCat(n) {
        if (I18N.locale === 'en' && typeof Intl !== 'undefined' && Intl.PluralRules) {
            try {
                return new Intl.PluralRules('en').select(Number(n));
            } catch (e) { /* fall through */ }
        }
        return 'other';
    }

    function fillParams(template, params, isHtml) {
        if (!params) return template;
        return String(template).replace(/\{(\w+)\}/g, function (m, name) {
            if (!Object.prototype.hasOwnProperty.call(params, name)) return m;
            var v = params[name];
            if (v == null) return '';
            return isHtml ? String(v) : escapeHtml(String(v));
        });
    }

    var I18N = {
        locale: DEFAULT_LOCALE,
        serverLocale: DEFAULT_LOCALE,
        dicts: { vi: {}, en: {} },
        SUPPORTED: SUPPORTED,
        DEFAULT_LOCALE: DEFAULT_LOCALE,

        t: function (key, params) {
            var value = lookup(I18N.locale, key);
            if (value === null) return key;
            return fillParams(value, params || null, /_html$/.test(key));
        },

        tp: function (base, n, params) {
            var cat = pluralCat(Number(n));
            var key = base + '_' + cat;
            var value = lookup(I18N.locale, key);
            if (value === null) value = lookup(I18N.locale, base + '_other');
            if (value === null) value = base + '_' + cat;
            // The count doubles as the {n} parameter unless overridden.
            var p = params ? Object.assign({}, params) : {};
            if (!Object.prototype.hasOwnProperty.call(p, 'n')) p.n = n;
            return fillParams(value, p, /_html$/.test(key));
        },

        detectBrowserLocale: function () {
            var lang = String(navigator.language || navigator.userLanguage || '').toLowerCase();
            if (lang.indexOf('vi') === 0) return 'vi';
            if (lang.indexOf('en') === 0) return 'en';
            return DEFAULT_LOCALE;
        },

        setLocale: function (loc, persist) {
            I18N.locale = SUPPORTED.indexOf(loc) >= 0 ? loc : DEFAULT_LOCALE;
            if (document.documentElement) {
                document.documentElement.setAttribute('lang', I18N.locale);
            }
            if (persist !== false) {
                try { localStorage.setItem(STORAGE_KEY, I18N.locale); } catch (e) { /* private mode */ }
                I18N.setCookie(I18N.locale);
            }
        },

        setCookie: function (loc) {
            try {
                document.cookie = COOKIE_KEY + '=' + encodeURIComponent(loc) +
                    '; path=/; max-age=31536000';
            } catch (e) { /* cookies disabled */ }
        },

        getCookie: function () {
            var m = document.cookie.match(new RegExp('(?:^|; )' + COOKIE_KEY + '=([^;]*)'));
            return m ? decodeURIComponent(m[1]) : '';
        },

        reloadGuard: function () {
            try {
                if (sessionStorage.getItem(RELOAD_GUARD)) return true;
                sessionStorage.setItem(RELOAD_GUARD, '1');
            } catch (e) {
                return true; // storage unavailable -> never loop
            }
            return false;
        },

        reloadTo: function (loc) {
            I18N.setLocale(loc, true);
            if (I18N.reloadGuard()) return;
            window.location.reload();
        },

        init: function () {
            var saved = '';
            try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }

            if (SUPPORTED.indexOf(saved) >= 0) {
                // User choice wins over everything (A0.3).
                if (saved !== I18N.serverLocale) {
                    // Server rendered another locale: persist + reload once.
                    I18N.reloadTo(saved);
                    return;
                }
                // Server already rendered the right locale: just sync the cookie.
                if (!I18N.getCookie()) I18N.setCookie(saved);
                I18N.setLocale(saved, false);
                return;
            }

            // First visit: detect from the browser language.
            var detected = I18N.detectBrowserLocale();
            try { localStorage.setItem(STORAGE_KEY, detected); } catch (e) { /* ignore */ }
            I18N.setCookie(detected);
            if (detected !== I18N.serverLocale) {
                I18N.reloadTo(detected);
                return;
            }
            I18N.setLocale(detected, false);
        },

        refreshCallbacks: [],
        onRefresh: function (fn) {
            if (typeof fn === 'function') I18N.refreshCallbacks.push(fn);
        },
        notifyRefresh: function () {
            I18N.refreshCallbacks.forEach(function (fn) {
                try { fn(); } catch (e) { /* one broken callback must not break the rest */ }
            });
            try {
                document.dispatchEvent(new CustomEvent('i18n:changed', {
                    detail: { locale: I18N.locale }
                }));
            } catch (e) { /* CustomEvent unsupported (ancient) */ }
        },

        injectDropdown: function () {
            if (document.getElementById('locale-switch')) return;
            var host = document.querySelector('.corr-topbar') || document.querySelector('header');
            if (!host) return;

            var select = document.createElement('select');
            select.id = 'locale-switch';
            select.className = 'locale-switch';
            select.setAttribute('aria-label', I18N.t('common.languageAria'));
            SUPPORTED.forEach(function (loc) {
                var opt = document.createElement('option');
                opt.value = loc;
                opt.textContent = LOCALE_LABELS[loc] || loc;
                if (loc === I18N.locale) opt.selected = true;
                select.appendChild(opt);
            });
            select.addEventListener('change', function () {
                var loc = select.value;
                // P1: let JS-driven strings re-render (hooks) before the reload
                // lands the fully-localized page (P0 flow).
                I18N.setLocale(loc, true);
                I18N.notifyRefresh();
                // F1 (UI gate): a language switch is explicit user intent and
                // must ALWAYS reload. The sessionStorage guard only protects the
                // automatic first-visit detection (init/reloadTo) against reload
                // loops — clear it here so a second switch is never swallowed,
                // which would mix the old static locale with the new JS locale.
                try { sessionStorage.removeItem(RELOAD_GUARD); } catch (e) { /* ignore */ }
                window.location.reload();
            });
            host.appendChild(select);
            I18N.injectStyles();
        },

        injectStyles: function () {
            if (document.getElementById('i18n-styles')) return;
            var style = document.createElement('style');
            style.id = 'i18n-styles';
            style.textContent =
                ".locale-switch{font-family:'Exo 2',sans-serif;font-size:13px;font-weight:500;" +
                'color:#5E1675;background:#fff;border:1px solid #5E1675;border-radius:8px;' +
                'padding:4px 8px;cursor:pointer;outline:none;max-width:160px;}' +
                '.locale-switch:hover{border-color:#4a1160;}' +
                '.locale-switch:focus-visible{box-shadow:0 0 0 2px rgba(94,22,117,0.25);}' +
                'header{display:flex;align-items:center;justify-content:flex-end;}' +
                'header .locale-switch{margin:2px 1.4% 2px 0;flex-shrink:0;}' +
                '.corr-topbar .locale-switch{margin-left:16px;flex-shrink:0;}' +
                '@media (max-width:600px){header .locale-switch{font-size:12px;padding:4px 6px;}}';
            document.head.appendChild(style);
        }
    };

    // ---- Boot ----
    I18N.dicts = parseDicts();
    var langAttr = document.documentElement.getAttribute('lang') || '';
    if (SUPPORTED.indexOf(langAttr) >= 0) I18N.serverLocale = langAttr;
    I18N.locale = I18N.serverLocale;
    I18N.init();

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', I18N.injectDropdown);
    } else {
        I18N.injectDropdown();
    }

    window.I18N = I18N;
})();
