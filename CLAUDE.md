# Zhulong (烛龙)

A Racket-hosted large-model system: Racket owns model definition, symbolic
optimization, and orchestration; heavy compute is delegated to libtorch and
AOTInductor artifacts.

- Design docs: debug/zhulong-design-zh.md (Chinese) and
  debug/zhulong-design-en.md (English) — keep the two in sync when editing.
- Implementation plan: debug/zhulong-impl-plan-zh.md (build order, module
  map). Architecture changes go to the design docs first, then code.
- Coding rules: STYLE.md at the repo root is canonical. Follow it strictly.

## Layout

Two linked single-collection packages: src/ (the framework; its collection
name is `zhulong`, set by info.rkt and pinned with `--name` at install) and
models/ (application-level model library). Requires use `zhulong/...` paths
regardless of the src/ directory name. Tests live in tests/ as
`<module>-test.rkt`. Dev tools live in tools/ (not shipped).

## Commands

- `make setup` — link-install the zhulong and models packages (once).
- `make test`  — compile and run all test suites.
- `make fmt`   — format all .rkt files with raco fmt (config: .fmt.rkt).
