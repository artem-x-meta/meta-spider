# meta-spider — docs site

Self-contained static documentation page (Django/readthedocs-style). No build, no server:
**just open `index.html` in any browser** (works from `file://`).

## Structure
- `index.html` — everything (HTML + embedded CSS + JS), single file.
- Layout: fixed header (brand, section filter, EN/RU toggle) · left sidebar nav · content.
- JS: section filter, active-section highlight on scroll, mobile menu, language toggle
  (persisted in `localStorage`).

## Content
- **EN** is complete (authored from the actual framework: pipeline, two-pass mechanism,
  components, config, training, evaluation, llama.cpp deploy, API, FAQ).
- All EN content lives inside `<div class="en"> … </div>` as `<section id="…">` blocks.
- The sidebar links (`#anchor`) map 1:1 to those section ids.

## Adding the RU translation (later, via Codex)
RU is currently a placeholder (`<div class="ru-stub">`). To add a real RU version:
1. Duplicate the `<div class="en">…</div>` block as `<div class="ru-content">…</div>`,
   translate the text inside (keep the `id`s and code blocks unchanged).
2. In the CSS, mirror the toggle rules: `body.ru main .en{display:none}` already hides EN;
   add `body:not(.ru) main .ru-content{display:none}` and remove/repurpose `.ru-stub`.
3. The toggle (`setLang`) already flips `body.ru` — no JS change needed.

Keep section `id`s identical across languages so the sidebar nav works for both.
