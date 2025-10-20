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
All datasets use local COCO 2017 images from `/data/coco/`:
- **POPE**: `lmms-lab/POPE` (adversarial/random/popular categories)
- **RefCOCO/+/g**: `lmms-lab/RefCOCO*` variants
- **COCO Caption**: Local files or HuggingFace

## Methods

### Standard Calibration
- **ERW**: Evidence Re-Weighting using visual grounding `g_i(t)`
- **PVA**: Persistent Visual Anchors with EMA `g̃_i(t)`
- **VEN**: Visual Evidence Normalization using ratio `r_i(t)`

Hyperparameters: `lambda_` (logit scale), `beta` (PVA momentum), `alpha` (PVA/VEN blend), `ven_eps`

### Extreme Logit Manipulation

**새로운 극단적 로짓 조작 방법들** - 시각적 근거가 있는 토큰을 극적으로 강화하고 환각 토큰을 억제:

1. **Contrastive Logit Sharpening** (`use_contrastive=true`)
   - Signal 기반으로 토큰을 두 그룹으로 분리
   - 시각적 토큰은 `contrastive_strength` 배로 강화
   - 환각 토큰은 `1/contrastive_strength` 배로 억제
   - Parameters: `contrastive_strength=10.0`, `contrastive_threshold=0.5`

2. **Adversarial Perturbation** (`use_adversarial=true`)
   - 반복적으로 확률 분포를 교란하고 재조정
   - Signal 낮은 고확률 토큰을 강력히 억제
   - Signal 높은 토큰을 지속적으로 강화
   - Parameters: `adversarial_strength=0.3`, `adversarial_iterations=3`

3. **Adaptive Temperature** (`use_adaptive_temp=true`)
   - 토큰별로 다른 temperature 적용
   - 높은 signal = 낮은 temp (sharper distribution)
   - 낮은 signal = 높은 temp (flatter distribution)
   - Parameters: `temp_base=1.0`, `temp_range=5.0`, `temp_visual_boost=true`

**Usage**:
```bash
# Contrastive only
python -m svlm.runner method.use_contrastive=true method.contrastive_strength=15.0

# All three methods combined
python -m svlm.runner \
  method.use_contrastive=true \
  method.use_adversarial=true \
  method.use_adaptive_temp=true

# With W&B sweeps (automatically searches extreme parameters)
./run_sweep.sh bayes 30
```

## Evaluation Metrics

| Metric | Status | Notes |
|--------|--------|-------|
| **POPE** | ✅ Complete | Accuracy, Precision, Recall, F1, Yes-rate |
| **CIDEr** | ✅ Complete | Requires `pip install pycocoevalcap` |
| **CHAIR** | ✅ Complete | CHAIRi & CHAIRs with COCO object extraction |
| **RefCOCO** | ✅ Complete | IoU-based Acc@0.5, @0.75, @0.9 with bbox parsing |

### Usage Notes

**CHAIR**: Extracts objects from captions using pattern matching against COCO 80 categories. For best results, provide ground truth objects in reference field. Supports spacy for advanced NLP (optional: `pip install spacy && python -m spacy download en_core_web_sm`).

**RefCOCO**: Parses bounding boxes from model output. Supports formats: `[x, y, w, h]`, `<box>x1, y1, x2, y2</box>`, or space-separated numbers. Ground truth bbox should be in reference field as list or dict.

## Hyperparameter Search with W&B Sweeps

Install wandb:
```bash
pip install wandb
wandb login
```

### Quick Start

```bash
# Bayesian optimization (recommended)
./run_sweep.sh bayes 20

# Grid search
./run_sweep.sh grid

# Random search
./run_sweep.sh random 50
```

### Manual Control

```bash
# Create sweep
wandb sweep configs/sweep.yaml

# Run agent (use sweep ID from above)
wandb agent YOUR_USERNAME/svlm-calibration/SWEEP_ID
```

### Sweep Configurations

Three sweep configs available in `configs/`:

| File | Method | Parameters | Best For |
|------|--------|------------|----------|
| `sweep.yaml` | Bayesian | lambda, beta, alpha, ven_eps, temp, top_p, **extreme params** | Finding optimal values |
| `sweep_grid.yaml` | Grid | lambda [0.3-0.9], beta [0.7-0.9], alpha [0.4-0.8], **extreme modes** | Systematic exploration |
| `sweep_random.yaml` | Random | lambda, beta, alpha, **extreme params (wide range)** | Quick initial search |

**All sweep configs now include extreme logit manipulation parameters**:
- `use_contrastive`, `use_adversarial`, `use_adaptive_temp` (boolean toggles)
- `contrastive_strength`, `adversarial_strength`, `adversarial_iterations`, `temp_range`, etc.

Modify configs to change search space or add parameters.
