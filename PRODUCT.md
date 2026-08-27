# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are manga readers, fan translators, and creators who need to turn one or many source-language manga pages into readable translated images without moving between separate OCR, translation, and typesetting tools. The exact audience priority is inferred from the explicit project brief and existing workflow.

## Product Purpose

Manga Translator turns uploaded manga/comic images into translated, rendered images through one workflow: recognize text with OCR, translate it, erase the source text, render the translation, and optionally correct speech bubbles manually. Success means a user can understand the workflow immediately, configure it confidently, and reach translated output without losing control of language, translator, style, or font choices.

## Positioning

The product combines OCR, translation, source-text cleanup, text rendering, and a WYSIWYG/manual correction path in one browser-based task rather than stopping at plain translated text.

## Operating Context

Users work with batches of JPG, PNG, WebP, BMP, TIFF, or AVIF manga/comic pages. They choose a source language, target language, translation provider, translation style, and lettering font, with optional Gemini or local-LLM settings. The homepage is both the product introduction and the task entry point.

## Capabilities and Constraints

- Flask, Jinja, and vanilla JavaScript; no frontend framework.
- Vietnamese and English localization with auto-detection and a locale selector.
- Existing DOM IDs/classes are behavioral contracts and must remain stable.
- The shared stylesheet also serves the translation-results page.
- User-supplied Gemini keys are stored in browser localStorage.
- No performance, customer, accuracy, or licensing claims may be invented.

## Brand Commitments

The product name is Manga Translator. The binding brand color is purple #5E1675 and the binding typeface is Exo 2. Existing logo asset: static/img/logo.png.

## Evidence on Hand

The runnable application, templates, current copy, workflow controls, logo, and results UI are the available proof. No testimonials, benchmark data, customer logos, or accuracy claims are available and must not be fabricated.

## Product Principles

- Make the end-to-end image transformation understandable before asking for configuration.
- Keep the three-step task path obvious and low-friction.
- Express manga culture through disciplined composition, not decorative anime clichés.
- Preserve user control over OCR correction, translation provider, style, and lettering.
- Keep localization, accessibility, and responsive behavior first-class.

## Accessibility & Inclusion

Meet WCAG 2.1 AA expectations: keyboard-operable controls, visible focus, sufficient contrast, reduced-motion support, semantic landmarks, and localized text.
