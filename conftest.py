"""
Root pytest conftest — empty on purpose. Its only job is to mark this directory as pytest's
rootdir so `modules`, `scripts`, `main`, `register_person`, etc. (all repo-root-relative, no
package installed) resolve the same way whether tests are invoked as `pytest project_tests/...` or
`python -m pytest project_tests/...`, regardless of which subdirectory under project_tests/ a given test file
lives in (see docs/architecture.md's repository layout — this project has no src/ layout or
installed package, so tests rely on repo-root-relative imports).
"""
