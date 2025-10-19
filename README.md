# SVLM Post-hoc Calibration Framework

This project implements a Hydra-driven experimentation framework for mitigating hallucinations in small vision-language models (VLMs) via post-hoc logit calibration. The framework exposes the three attenuation methods defined in the spec—Evidence Re-Weighting (ERW), Persistent Visual Anchors (PVA), and Visual Evidence Normalization (VEN)—and allows them to be toggled and combined without altering model parameters.

## Features
- Step-wise calibration that only uses cross-modal and self-attention tensors collected during generation.
- Modular Hydra configuration for models, datasets, methods, decoding, and run metadata.
- JSON logging of runs, including configuration, per-step diagnostics, aggregate metrics, and Δ improvements over a baseline run when available.
- Pluggable adapter interface so you can integrate any VLM that provides logits and attention matrices per decoding step.

## Project Layout
```
configs/                     # Hydra configuration groups
src/svlm/                    # Core library code
  attention.py               # g_i(t), l_i(t), persistence & ratio computations
  calibrator.py              # LogitCalibrator with ERW, PVA, VEN combination logic
  config_schema.py           # Dataclass definitions backing Hydra configs
  evaluation.py              # Metric stubs (POPE, CHAIR, COCO/CIDEr, RefCOCO+)
  model_adapter.py           # Adapter protocol + DummyAdapter
  pipeline.py                # Autoregressive loop, sampling, metadata export
  runner.py                  # Hydra entrypoint
```

## Quick Start
1. **Install dependencies**
   ```bash
   pip install -e .
   ```
   This brings in `hydra-core`, `torch`, `tqdm`, `numpy`, and `pydantic`. Install any model-specific packages (e.g., `transformers`) separately if you plan to integrate a real VLM.

2. **Run the baseline (no calibration)**
   ```bash
   python -m svlm.runner method=baseline run=baseline
   ```

3. **Enable a calibration method**
   ```bash
   python -m svlm.runner method=erw_only run=erw_only
   python -m svlm.runner method=erw_pva run=erw_pva
   python -m svlm.runner method=erw_ven run=erw_ven
   python -m svlm.runner method=erw_pva_ven run=erw_pva_ven
   ```

4. **Override configs on the fly**
   ```bash
   python -m svlm.runner model=minicpm_v decoding.max_new_tokens=32 method.lambda_=0.45
   python -m svlm.runner model.gpu_id=1 data.data_path=/datasets/pope
   ```

Each run writes metadata to the directory specified in `run.output_dir`, including per-step diagnostic stats, evaluation metrics (placeholder implementations), and Δ against a baseline metadata file if available.

## Integrating a Real VLM
The default configuration uses the `DummyAdapter`, which fabricates logits and attention tensors for smoke testing. To hook up a genuine model:

1. Implement the `VisionLanguageAdapter` protocol defined in `svlm/model_adapter.py`:
   - `prepare_inputs(sample)` should tokenize the prompt, encode the visual inputs, and return an `AdapterState`.
   - `forward(state)` must return a `StepOutput` containing:
     - `logits`: a 1D tensor for the next token distribution.
     - `attention`: an `AttentionProjections` object with cross-modal `[H, N_v, N_t]` and (optional) self-attention `[H, N_t, N_t]` tensors.
     - `past_key_values`: cached model state for efficient autoregressive decoding.
   - `update(state, token_id)` should append the sampled token and advance the cache.
   - `decode(token_ids)` maps generated token IDs to text.

2. Register the adapter via Hydra by overriding `model.adapter_target` and `model.adapter_kwargs`. Set `adapter_target=auto` (default) to let the runner pick an adapter based on the model name or fall back to the dummy adapter. To point at a custom implementation:
   ```bash
   python -m svlm.runner \
     model.adapter_target=your_package.adapters.QwenAdapter \
     model.adapter_kwargs='{"model_name": "Qwen/Qwen2-VL-2B-Instruct"}'
   ```

3. Ensure the adapter sets `output_attentions=True` and captures cross-modal + self-attention tensors per decoding step so the calibrator can compute `g_i(t)`, `g̃_i(t)`, and `r_i(t)` precisely as defined in the spec.

Use `model.gpu_id=<int>` to pin inference to a specific CUDA device, and `data.data_path=/path/to/eval` to direct dataset loaders toward concrete evaluation assets.

## Evaluation Stub
`src/svlm/evaluation.py` provides lightweight stand-ins for:
- **POPE:** accuracy, yes-rate, and an F1 placeholder based on prefix matches.
- **CHAIR / COCO CIDEr / RefCOCO+:** return zeros by default. Replace with task-specific evaluators as you integrate real datasets.

When a baseline metadata file exists (default: `outputs/baseline/metadata.json`), subsequent runs automatically compute Δ metrics and log them alongside the current results.

## Next Steps
- Replace `DummyAdapter` with adapters for the target VLMs (Qwen2-VL-2B, MiniCPM-V, etc.).
- Swap in real dataset loaders inside `collect_samples` or route through Hydra to dataset-specific readers.
- Implement true metric calculators for POPE, CHAIR, CIDEr, and RefCOCO+ once annotations are available.
- Add automated tests (e.g., via `pytest`) to validate the calibration math and configuration wiring.
