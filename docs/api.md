# Vita AI – API Reference

Base URL: `http://localhost:8000`

## Endpoints

### Health & Info

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Project info and version |
| `GET` | `/health` | Health check |
| `GET` | `/models/status` | All model statuses |

### Predictions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict/face` | Heart rate from video upload |
| `POST` | `/predict/audio` | Breathing analysis from audio upload |
| `POST` | `/predict/symptom` | Symptom risk from text |
| `POST` | `/predict/final-score` | Combined Vita Health Score |

### User Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/signup` | Create user account |
| `POST` | `/auth/login` | Login |
| `GET` | `/user/{user_id}/health-data` | Get health data |
| `POST` | `/user/{user_id}/health-data` | Submit health data |
| `GET` | `/user/{user_id}/scan-results` | Get scan results |

### WebSocket

| Path | Description |
|------|-------------|
| `WS /ws/stream` | Real-time streaming analysis |

## Request/Response Examples

### POST /predict/face
```
Content-Type: multipart/form-data
Body: file=<video_file>

Response: { "heart_rate": 72.5, "confidence": 0.85, ... }
```

### POST /predict/symptom
```json
{ "text": "I have a headache and fever" }

Response: { "symptoms": [...], "risk_level": "moderate", ... }
```

### POST /predict/final-score
```json
{
  "face_score": 0.8,
  "audio_score": 0.7,
  "symptom_score": 0.6
}

Response: { "vita_score": 75.2, "risk_level": "low", ... }
```
