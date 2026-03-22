# Sample Data

Place your test media files here for local development and demos.

## Expected files

| File | Purpose | Format |
|---|---|---|
| `sample_face_video.mp4` | Short (5-15 s) face video for heart-rate estimation | `.mp4` / `.avi` |
| `sample_breathing.wav` | Short (10-30 s) breathing audio recording | `.wav` / `.mp3` |

## How to record

### Face video
- Use your laptop/phone camera.
- Film your face straight-on, in decent lighting, for ~10 seconds.
- Keep still — minimal head movement improves rPPG accuracy.

### Breathing audio
- Use any voice recorder app.
- Breathe naturally (nose or mouth) for ~15-20 seconds.
- A quiet environment produces cleaner results.

## Notes
- These files are **not** committed to version control (see `.gitignore`).
- The API and tests work without these files (they use fallback/mock paths).
