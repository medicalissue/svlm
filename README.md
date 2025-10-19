# SVLM Post-hoc Calibration

Post-hoc logit calibration framework for mitigating hallucinations in vision-language models using ERW, PVA, and VEN methods.

## Setup
```bash
pip install -e .
```

## Usage
```bash
# Run with baseline (no calibration)
python -m svlm.runner method=baseline run=baseline

# Enable calibration methods
python -m svlm.runner method=erw_pva run=erw_pva
python -m svlm.runner method=erw_pva_ven run=erw_pva_ven

# Override configs
python -m svlm.runner model=minicpm_v decoding.max_new_tokens=32 method.lambda_=0.45
```

## Structure
- `configs/` - Hydra config groups (data, model, method, decoding, run)
- `src/svlm/` - Core library
  - `runner.py` - Entry point
  - `pipeline.py` - Data loading & generation loop
  - `calibrator.py` - ERW/PVA/VEN signal combination
  - `attention.py` - Attention signal computations
  - `model_adapter.py` - VLM adapter protocol
  - `evaluation.py` - Metrics (POPE, CHAIR, RefCOCO+)

## Datasets
Configured via `configs/data/`:
- **POPE**: `lmms-lab/POPE` (adversarial/random/popular splits)
- **RefCOCO+**: `lmms-lab/RefCOCO_Grounding`
- **COCO-Caption**: Local COCO files or `lmms-lab/COCO-Caption`

## Methods
- **ERW**: Evidence Re-Weighting using visual grounding `g_i(t)`
- **PVA**: Persistent Visual Anchors with EMA `g̃_i(t)`
- **VEN**: Visual Evidence Normalization using ratio `r_i(t)`

Hyperparameters: `lambda_` (logit scale), `beta` (PVA momentum), `alpha` (PVA/VEN blend), `ven_eps`
