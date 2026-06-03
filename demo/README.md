# /demo — Interactive Web Demo

This folder contains the interactive web-based demonstration system (React frontend + Python backend).

**Will contain:**
- `backend/` — FastAPI server (REST API + WebSocket for real-time streaming, debate orchestration endpoints)
- `frontend/` — React + TypeScript + TailwindCSS application with 5 core features:
  1. Topic selection (4 German policy topics)
  2. Strategy/condition selection (4 strategies + 4 baselines)
  3. Moderator rationale display (expandable trigger details)
  4. Side-by-side strategy comparison (split-screen, same topic)
  5. Manual intervention injection (user-authored moderator messages)
- Language toggle (German/English)
- Deployment configuration
