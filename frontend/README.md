# FinSlang-AI Dashboard

A premium Indonesian financial sentiment analysis dashboard built with React, Vite, Tailwind CSS, Lucide icons, and Recharts.

## Quick Start

1. Open a terminal in `frontend`
2. Install dependencies:
```bash
npm install
```
3. Start your FastAPI backend on port 8000 from the repository root:
```bash
python AOL_NLP/app.py
```
4. Run the frontend development server:
```bash
npm run dev
```
5. Open the local URL shown in the terminal.

The frontend fetches predictions from `/predict/multi` and requires the FastAPI backend to be running.

## Features

- Dark premium fintech UI
- Central sentiment text input
- Ensemble consensus result card
- Circular gauge charts for deep learning models
- Horizontal baseline comparison for traditional models
- Clean, focused dashboard layout

## Notes

The dashboard attempts to fetch from `/predict/multi`. If the backend is unavailable, it displays demo data instead.
