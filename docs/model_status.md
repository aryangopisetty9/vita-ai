# Vita AI – Model Status

## Integrated Models

| Model | Category | Status | Auto-Download | Location |
|-------|----------|--------|---------------|----------|
| DistilBERT | NLP | Verified | Yes | `backend/data/models_cache/distilbert/` |
| BioBERT | NLP | Verified | Yes | `backend/data/models_cache/biobert/` |
| YAMNet | Audio | Verified | Yes | `backend/data/models_cache/yamnet_saved_model/` |
| VGGish | Audio | Verified | Manual | `backend/data/models_cache/vggish/` |
| open-rPPG | Face/rPPG | Verified | pip-installed | Bundled with `open-rppg` package |
| Fusion MLP | Fusion | Built-in | N/A | In-memory (no weights file) |

## Manual-Download Models (rPPG)

| Model | Status | Weight File |
|-------|--------|-------------|
| EfficientPhys | Available | `backend/data/models_cache/efficientphys/*.pth` |
| PhysFormer | Available | `backend/data/models_cache/physformer/*.pth` |
| FacePhys | Available | `backend/data/models_cache/facephys/*.pth` |
| PhysMamba | Available | `backend/data/models_cache/physmamba/*.pth` |

## Legacy Models (Deprecated)

| Model | Status | Notes |
|-------|--------|-------|
| TS-CAN | Deprecated | Replaced by open-rPPG |
| PhysNet | Deprecated | Replaced by open-rPPG |
| DeepPhys | Deprecated | Replaced by open-rPPG |

## Checking Status

```bash
# Via API
curl http://localhost:8000/models/status

# Via Python
python -c "from backend.app.ml.registry.model_status import get_all_model_status; print(get_all_model_status())"
```
