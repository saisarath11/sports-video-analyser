# AI Sports Video Analyser 🏅

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00FF00)](https://mediapipe.dev/)
[![Gemini](https://img.shields.io/badge/Google-Gemini_AI-8E44AD)](https://aistudio.google.com/)

A complete, end-to-end **AI-powered Sports Video Analysis Platform** built with Computer Vision, Deep Learning, Object Detection, Multi-Object Tracking, Pose Estimation, and Gemini LLM Coaching Feedback.

---

## 🌟 Features

- **Video Preprocessing & Sampling**: OpenCV video loading, metadata extraction, and intelligent frame sampling (FAST, BALANCED, DETAILED modes).
- **Player Detection**: YOLOv8-based person detection with bounding boxes and confidence scores.
- **Multi-Player Tracking**: Persistent Multi-Object Tracking (ByteTrack) assigning unique player IDs across video frames.
- **Body Posture Analysis**: MediaPipe Pose 33-landmark estimation and joint angle calculation (knees, hips, elbows, trunk lean).
- **Kinematics & Motion Metrics**: Total distance (px & estimated metres), average/peak speed (px/s & estimated km/h), active/stationary time ratios, movement count, and direction changes.
- **Spatial Visualizations**: Trajectory mapping and position heatmaps.
- **Modular Event Detection Engine**: Rule-based sports event detection supporting **Football, Cricket, Basketball, Badminton, and Tennis**.
- **Performance Evaluation Engine**: Prototype performance indicator scoring (0–100) and letter grading (A+ to E).
- **Strengths & Weaknesses Analysis**: Automated identification of player strengths and areas needing improvement.
- **Personalized AI Coaching Feedback**: Integration with Google Gemini API for personalized coaching advice, with template fallback.
- **Interactive Streamlit Dashboard**: Multi-page dashboard with Plotly charts, metric cards, and video player controls.
- **Annotated Output Video**: Rendered MP4 output with player bounding boxes, IDs, pose skeletons, and trajectory trails.
- **Downloadable PDF Reports & Export**: FPDF2-generated PDF reports, raw CSV, and JSON data exports.

---

## 🏆 Supported Sports

| Sport | Custom Events Detected | Key Focus Area |
|---|---|---|
| ⚽ **Football (Soccer)** | Running, Sprinting, Sprint Burst, Defensive Positioning | Pitch coverage, high-speed sprints, work rate |
| 🏏 **Cricket** | Batting Stance, Bowling Action, Running between wickets, Fielding | Stance balance, reaction speed, acceleration |
| 🏀 **Basketball** | Defensive Stance, Jump Attempt, Dribbling Movement, Transition | Court coverage, lateral shuffle, transition speed |
| 🏸 **Badminton** | Ready Position, Net Approach, Jump Smash Prep, Baseline Return | Center recovery speed, split-step footwork |
| 🎾 **Tennis** | Baseline Movement, Net Approach, Lateral Shuffle, Serve Prep | Rally recovery, lateral crossover steps |

---

## 🚀 Installation & Setup

### 1. Clone / Navigate to Project Folder
```bash
cd C:\Users\saisa\OneDrive\Desktop\sports-video-analyser
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Open `.env` and add your **Google Gemini API Key** (optional for LLM feedback):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🏃 Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Open your web browser at `http://localhost:8501`.

---

## 📁 Project Architecture

```
sports-video-analyser/
│
├── app.py                          ← Streamlit Home Page
├── requirements.txt                ← Python Dependencies
├── README.md                       ← Documentation
├── .env.example                    ← Configuration Template
├── .gitignore
│
├── sports/                         ← Sport Configuration Layer
│   ├── __init__.py
│   └── config.py                   ← Multi-sport rules & thresholds
│
├── core/                           ← AI/ML Pipeline Engine
│   ├── __init__.py
│   ├── models.py                   ← Shared Data Models
│   ├── video_processor.py          ← OpenCV Video Loader & Sampler
│   ├── detector.py                 ← YOLOv8 Player Detector
│   ├── tracker.py                  ← ByteTrack Object Tracker
│   ├── pose_estimator.py           ← MediaPipe 33-Landmark Pose Estimator
│   ├── motion_analyzer.py          ← Kinematics & Motion Engine
│   ├── event_detector.py           ← Modular Activity Event Engine
│   ├── performance_analyzer.py     ← Performance Indicator Evaluator
│   └── feedback_generator.py       ← Gemini AI Feedback Generator
│
├── visualization/                  ← Dashboard & Report Rendering
│   ├── __init__.py
│   ├── charts.py                   ← Plotly Interactive Visualizers
│   ├── heatmap.py                  ← Matplotlib Position Heatmaps
│   ├── trajectory.py               ← 2D Trajectory Path Visualizers
│   ├── annotator.py                ← OpenCV Video Annotator
│   └── report.py                   ← PDF Report Generator (FPDF2)
│
├── utils/                          ← Helper Utilities
│   ├── __init__.py
│   ├── config.py                   ← App Settings
│   ├── file_utils.py               ← File Validation & Storage
│   ├── db.py                       ← SQLite Session Database
│   └── logging_utils.py            ← Logger Setup
│
├── pages/                          ← Multi-Page Streamlit App
│   ├── 01_upload.py
│   ├── 02_analysis.py
│   ├── 03_players.py
│   ├── 04_movement.py
│   ├── 05_events.py
│   ├── 06_performance.py
│   ├── 07_feedback.py
│   └── 08_report.py
│
└── tests/                          ← Automated Unit Tests
    ├── test_detector.py
    ├── test_tracker.py
    ├── test_motion.py
    └── test_feedback.py
```

---

## 🧪 Running Unit Tests

Run test suites using `pytest`:
```bash
pytest tests/ -v
```

---

## ⚠️ Disclaimers

1. **Prototype Indicators**: Performance scores (0–100) are prototype indicators generated by AI algorithms based on video observations. They are not scientifically or medically validated.
2. **Kinematic Estimates**: Real-world distance (metres) and speed (km/h) metrics are derived from pixel measurements and calibrated heuristics. They are clearly labelled as estimates.
3. **No Medical Diagnosis**: Posture observations and recommendations are intended solely for sports performance demonstration and do not constitute medical or biomechanical injury diagnosis.

---

## 📜 License

Distributed under the MIT License.
