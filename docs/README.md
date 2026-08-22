# Documentation Index

Reference material for this project, kept up to date with the codebase as it changes. Written to
be usable as source material for a written report — each file stands alone but cross-links to
the others.

| Doc | Covers |
|---|---|
| [`architecture.md`](architecture.md) | System overview, repository layout, cross-module design rules/conventions, pipeline flow diagrams (both `main.py` pipelines, stage-by-stage input/output), CLI usage, debug/visualization architecture |
| [`technologies.md`](technologies.md) | The concrete tech stack — libraries, every model used (what it is, where it's from, why it was chosen), storage formats |
| [`modules.md`](modules.md) | Per-module deep dive: purpose, working principle/algorithm, public contract (inputs/outputs), key parameters, known limitations, for all 11 modules |
| [`parameters.md`](parameters.md) | Every tunable value in `config/thresholds.yaml` — meaning, current value, calibration status (🔴 uncalibrated / 🟡 starting guess / 🟢 working default), tuning notes |
| [`commands.md`](commands.md) | Every command to run — the full pipeline (`main.py`), and each module's standalone `test_*.py`/`visualize_*.py` tools, run independently |
| [`overlay_colors.md`](overlay_colors.md) | Every debug-overlay color across every module — what it means, what's drawn, what it represents, including how colors are reused across layers when composited |

## Suggested reading order

1. **`architecture.md`** first — establishes the vocabulary (pipelines, stages, design rules)
   everything else assumes.
2. **`technologies.md`** for the "what's under the hood" inventory.
3. **`modules.md`** for how each stage actually works internally.
4. **`parameters.md`**, **`commands.md`**, and **`overlay_colors.md`** as references — the first
   while calibrating or reading `config/thresholds.yaml`, the second whenever you actually want
   to run something, the third whenever you're staring at a `--show --debug` window.

## Source of truth

These docs describe the code as of the last time they were updated — if something looks
inconsistent with `main.py`, `config/thresholds.yaml`, or a module's `interface.py`, the code
wins. See also `plans/01-04` for the original per-module design specs these modules were built
against (useful for *why* a decision was made, where this documentation set focuses on *what*
currently exists).
