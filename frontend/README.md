# Open-Set Recognition Using Underwater SONAR Images — Research Companion

This frontend web application is the interactive research companion for the Final Year Project:
**"Open-Set Recognition Using Underwater SONAR Images"**
Department of Computer Science, Bahria University, Islamabad.

**Authors:** Mohammad Khizar Hayat Cheema (01-134231-039), Muhammad Haroon (01-134231-052)  
**Supervisor:** Dr. Arif Ur Rahman

---

## Getting Started

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Development Server
To run the research companion locally:
```bash
npm run dev
```
The site will start at `http://localhost:5173/`.

### 3. Production Build
```bash
npm run build
npm run preview
```

---

## Live Threshold Explorer Configuration

The Threshold Explorer (Section 7) supports live per-sample threshold simulation.

- By default, it operates in **Verified Operating Points Mode**, accurately displaying the exact benchmark results from `outputs/results/all_results.json` along with empirical distribution charts.
- To unlock the **Live Per-Sample Histogram & Dynamic Recalculation Slider**, execute the exporter from the repository root:
  ```bash
  python scripts/export_scores.py
  ```
  This will load trained checkpoints from `outputs/models/`, run inference over the test partitions, and write `frontend/public/data/scores.json`. The web interface detects this file and unlocks real-time continuous histogram sliding and sample count updates.

---

## Interactive Features & Shortcuts

- **1 / 2 / 3**: Switch active OSR paradigm (1: MSP, 2: Deep SVDD, 3: Deep Metric Learning).
- **[ and ]**: Nudge the operational threshold slider up or down.
- **Esc**: Dismiss the full-screen figure Lightbox or shortcuts modal.
- **?**: Toggle the keyboard shortcut legend.
- **Dark / Light**: Toggle laboratory high-contrast color scheme (persisted to `localStorage`).
- **Figure Lightbox**: Click any acoustic figure to inspect high-resolution details with keyboard arrow navigation.
