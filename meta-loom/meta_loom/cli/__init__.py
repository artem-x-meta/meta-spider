"""metaloom CLI — thin stage commands on top of Meta-Loom.

`metaloom collect | train | eval | cloud` — each stage is a separate process with explicit
artifacts (dataset.pt / checkpoint.pt / report.json), tied together by the run.json manifest.
The stage logic reuses meta_core/meta_loom; the CLI only orchestrates + writes manifest/status.

Console-script entry point: `meta_loom.cli.main:main` (see pyproject [project.scripts]).
"""
