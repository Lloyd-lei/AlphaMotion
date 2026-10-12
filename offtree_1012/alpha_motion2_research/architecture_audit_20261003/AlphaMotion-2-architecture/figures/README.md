# Source-audited architecture figures

Final figure families:

| Subject | Compact paper file stem | Detailed reference stem |
|---|---|---|
| Three implemented systems | `paper_01_comparison` | `01_implemented_systems` |
| AlphaMotion-1 effective megatron model | `paper_02_alphamotion1` | `02_alphamotion1_layers` |
| UniMate Graph AdaLN | `paper_03_unimate` | `03_unimate_layers` |
| UMR learning and solver | `paper_04_umr` | `04_umr_layers` |
| AlphaMotion-2 proposed design | `paper_05_alphamotion2_proposed` | `05_alphamotion2_proposed` |

Each stem has editable `.svg`, vector `.pdf`, and `_preview.png`. Compact stems also have a 240 dpi `.png` for Google Docs insertion. Compact PDFs are 510 pt wide; body text is at least 8 pt and footer/source notes at least 7.1 pt at native physical size. Detailed references are 1140 pt wide and should be read at large size or with zoom, not directly shrunk into a paper column.

All 10 final PDF pages were rendered and visually reviewed. Text is selectable, there are no embedded raster images in the PDFs, and automated box-height checks report zero overflow. The same drawing primitives produce both PDF and SVG. See `layout_checks.json` and `pdf_structural_qa.json` for machine-readable checks.

`sources.md` maps every implemented-system source ID to exact code location and pinned public commits. `captions_zh.md` contains Chinese captions. Scientific interpretation and final research choices remain with the main architecture study. AlphaMotion-2 is explicitly proposed and unvalidated.

Rebuild using the bundled Python runtime:

```bash
/Users/yudonglei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 build_figures.py
```

Dependencies: ReportLab, Arial fonts on macOS, and Poppler `pdftoppm` on PATH.
