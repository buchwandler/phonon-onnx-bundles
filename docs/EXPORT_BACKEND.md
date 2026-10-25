# Export backend

The MVP uses a **pinned bootstrap recipe** rather than copying a large converter into this repository. `scripts/export_phonon.py` downloads the exact exporter revision declared in `registry/models.json`, resolves the selected Gradium model to an immutable Hugging Face commit, executes the recipe, then selects one stable four-graph layout.

The bundle ABI is:

- `onnx/text_conditioner.onnx`: token IDs + `duration_delta` -> text embeddings + BOS embedding
- `onnx/flow_lm_main.onnx`: input embeddings + KV cache -> hidden state + EOS logit + updated cache
- `onnx/flow_lm_flow.onnx`: hidden state + noise + `duration_delta` -> 32-d latent + next embedding
- `onnx/mimi_decoder.onnx`: streaming latent(s) + decoder state -> 24 kHz PCM + updated state

For q8/q4, only the two flow graphs use MatMulNBits; the text conditioner and Mimi decoder remain fp32. fp16 converts all graphs while keeping their public I/O float32.

A future native exporter can replace the bootstrap implementation as long as it emits this ABI and keeps `verify_candidate.py` passing.
