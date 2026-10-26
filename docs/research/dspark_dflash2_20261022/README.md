# DSpark / DFlash2 research evidence (2026-10-22)

Supporting material for the [AMD feasibility report](../../DSpark与DFlash2_替换DFlash可行性_双后端与开源方案评估_20261022.md ).

- `*_notes.md`: checkpoint, source, and public benchmark review notes.
- `hf_*metadata.json`: public configs, revisions, and bounded safetensors JSON headers; no tensor payloads.
- `read_hf_*metadata.py`: optional metadata collectors using the standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables. These collect the current public revisions when run; the saved JSON files preserve the research snapshots.

These materials document a static review, not an AMD GPU benchmark or a completed DSpark/DFlash2 integration. Model weights, local source clones, credentials, and inference logs are not included.
