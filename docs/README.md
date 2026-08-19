# UOG_AIS_FOLLOWME — Tech Report: Wave + Facing Trigger Gate Demo

Documentation set for the Quick Demo Spec implementation (person-tracking Follow-Me pipeline +
new wave-gesture/facing-camera trigger layer). Start here.

| Doc | Contents |
|---|---|
| [`architecture.md`](architecture.md) | Full system layout — existing Follow-Me pipeline (unmodified) + new demo layer, component responsibility table, class diagram, per-frame flow diagram. |
| [`technologies.md`](technologies.md) | Tech stack for both the existing pipeline and the new demo layer, with rationale for each new dependency. |
| [`parameters.md`](parameters.md) | Every tunable value in `config/settings.yaml`, what it controls, and the trade-off in each direction — including Raspberry Pi 5 tuning guidance. |
| [`master_spec.md`](master_spec.md) | Verbatim copy of the spec this work implements (`Project_Master_Doc.md`), with provenance notes. |
| [`implementation_audit.md`](implementation_audit.md) | What was built, every deviation from a literal spec reading (and why), what was tested and how, and what's still unverified. |
| [`diagrams/`](diagrams/) | Raw Mermaid sources for the two diagrams (also embedded in `architecture.md`). |

## Quick orientation

- **Real demo** (spec-compliant, requires a registered person): `python demo_wave_trigger.py`
- **Any-person test mode** (bypasses identity check, no registration needed):
  `python demo_wave_trigger.py --any-person --reacquisition-method position`
- **What's new vs. existing:** everything under `src/pose_estimator.py`, `src/wave_detector.py`,
  `src/any_person_tracker.py`, and `demo_wave_trigger.py` is new. `src/pipeline.py`,
  `src/detector.py`, `src/verifier.py`, and everything else under `src/` is unmodified — the demo
  only reads their output.
- **Every run writes a per-frame CSV log** (`wave_trigger_demo.log_csv_path`, default
  `logs/wave_trigger_demo_log.csv`) with trigger state and per-module timing — see
  `parameters.md`'s Pi 5 tuning section.
- **Read `implementation_audit.md` first** if you're about to calibrate the placeholder
  thresholds or extend this demo — it lists every judgment call made where the spec was silent
  or an underlying doc was missing.
