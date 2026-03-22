# Vita AI Backend

FastAPI-based health screening backend with modular ML pipelines for face/rPPG analysis, audio breathing detection, NLP symptom screening, and fusion scoring.

## Project Structure

```
backend/
├── app/                    # Main application package
│   ├── api/                # FastAPI routes, middleware, streaming
│   ├── core/               # Config, validation, exceptions, logging
│   ├── db/                 # Database models, schemas, engine
│   ├── ml/                 # Machine learning modules
│   │   ├── face/           # Face detection, rPPG, open-rPPG backend
│   │   ├── audio/          # Audio/breathing analysis (YAMNet, VGGish)
│   │   ├── nlp/            # NLP symptom analysis (DistilBERT, BioBERT)
│   │   ├── fusion/         # Score fusion engine
│   │   └── registry/       # Model registry, paths, download, status
│   ├── services/           # Session management
│   └── utils/              # Signal processing utilities
├── tests/                  # Test suite
├── eval/                   # Evaluation harness
├── train/                  # Training scripts
├── scripts/                # CLI utilities (model download)
├── data/                   # Runtime data (models_cache, DB, eval results)
└── sample_data/            # Sample inputs for testing
```

## Running

```bash
# From project root
uvicorn backend.app.api.main:app --reload

# Run tests
pytest backend/tests -q --tb=short
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITA_AUTO_DOWNLOAD_MODELS` | `false` | Auto-download NLP/audio models on startup |
| `VITA_ENABLE_DISTILBERT` | `true` | Enable DistilBERT sentiment model |
| `VITA_ENABLE_BIOBERT` | `true` | Enable BioBERT clinical model |
| `VITA_ENABLE_YAMNET` | `true` | Enable YAMNet audio classifier |
| `VITA_ENABLE_VGGISH` | `false` | Enable VGGish audio embeddings |
| `VITA_AUTO_DOWNLOAD_RPPG` | `false` | Auto-download rPPG model weights |
