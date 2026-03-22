# Vita AI Architecture

## Overview

Vita AI is a multi-modal health screening system that combines face/rPPG analysis, audio breathing detection, NLP symptom analysis, and fusion scoring into a unified health assessment.

## System Architecture

```
┌─────────────────────────────────────────────────┐
│                  Flutter Frontend                │
│              (lib/services/api_service)           │
└───────────────────────┬─────────────────────────┘
                        │ HTTP / WebSocket
┌───────────────────────▼─────────────────────────┐
│              FastAPI Application                 │
│           (backend/app/api/main.py)              │
├──────┬──────┬──────┬──────┬─────────────────────┤
│ Face │Audio │ NLP  │Fusion│  Model Registry      │
│Module│Module│Module│Engine│  & Status Tracker    │
└──────┴──────┴──────┴──────┴─────────────────────┘
         │         │         │
    ┌────▼────┐ ┌──▼──┐ ┌───▼───┐
    │open-rPPG│ │YAM- │ │Distil-│
    │TS-CAN   │ │Net  │ │BERT   │
    │PhysNet  │ │VGG- │ │Bio-   │
    │DeepPhys │ │ish  │ │BERT   │
    └─────────┘ └─────┘ └───────┘
```

## ML Pipeline

1. **Face Module** (`backend/app/ml/face/`): Video → face detection → rPPG → heart rate estimation
2. **Audio Module** (`backend/app/ml/audio/`): Audio → breathing pattern → risk classification
3. **NLP Module** (`backend/app/ml/nlp/`): Symptom text → sentiment + clinical analysis
4. **Fusion Engine** (`backend/app/ml/fusion/`): Combines all modality scores → Vita Health Score

## Data Flow

- Uploaded files are saved to `temp/` and cleaned up after processing
- Model weights are cached in `backend/data/models_cache/`
- Scan results can be persisted to SQLite via `backend/data/vita.db`
