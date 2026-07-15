"""metaloom CLI — thin stage commands on top of daimon-loom.

`metaloom collect | train | eval | cloud` — each stage is a separate process with explicit
artifacts (dataset.pt / checkpoint.pt / report.json), tied together by the run.json manifest.
The stage logic reuses meta_attention/daimon_loom; the CLI only orchestrates + writes manifest/status.

Console-script entry point: `daimon_loom.cli.main:main` (see pyproject [project.scripts]).
"""
