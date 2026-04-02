<div align="center">

# 🤖 RetinaFace Pro

### Face Detection · Landmark Localisation · Identity Verification

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-FFD21E?style=for-the-badge)](https://imdatascientistsachin-ratina-face.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> **Industry-grade computer vision** for dense face detection, precise 5-point landmark localisation, and one-to-many identity verification — production-ready and deployed live.

[**🚀 Try Live Demo**](https://imdatascientistsachin-ratina-face.hf.space) · [**📦 HuggingFace Space**](https://huggingface.co/spaces/ImdataScientistSachin/Ratina_face) · [**👤 Author**](#-author)

</div>

---

## 📖 Overview

**RetinaFace Pro** is a high-performance computer vision pipeline designed for dense face detection and identity verification. Unlike simple research scripts, this project is built as a **modular ML system** wrapping state-of-the-art models into a professional engineering architecture.

### 🚀 The Engineering Value-Add

- ✅ **Production Wrapper**: Clean, documented Python API (`src/detector.py`).
- ✅ **ML Monitoring**: Integrated **MLflow** tracking for inference latency and confidence drift.
- ✅ **Reliability**: 100% testable logic with **Pytest** and automated CI/CD.
- ✅ **Portability**: Fully Dockerized for zero-friction deployment on **Hugging Face Spaces**.

---

## 📊 Technical Benchmarks

_Tested on: NVIDIA GTX 1650 GPU | 16GB RAM | Intel i7_

| Task                      | Latency (ms) | Throughput (FPS) |
| :------------------------ | :----------- | :--------------- |
| Single Face Detection     | ~45ms        | ~22 FPS          |
| Multi-Face (5+) Detection | ~65ms        | ~15 FPS          |
| Identity Verification     | ~110ms       | ~9 FPS           |

---

## 🏗️ Architecture Decisions

To build a "Top Class" system, we prioritized **Engineering Trade-offs** over blind model selection. Here is the *Why* behind our choices:

### 1 · RetinaFace over Two-Stage Detectors
While two-stage detectors (Faster R-CNN) are accurate, they are often too slow for real-time inference. We chose **RetinaFace** because its **Feature Pyramid Network (FPN)** handles multi-scale faces (tiny to large) in a single pass, eliminating the need for a redundant Region Proposal Network (RPN) overhead.

### 2 · ArcFace for Verification Accuracy
Standard Softmax loss functions struggle with face verification because they don't optimize for embedding compactness. We implemented **ArcFace (Additive Angular Margin Loss)** because it enforces a tighter distance between similar faces in the hyperspace, leading to superior identity separability compared to standard cosine similarity.

### 3 · MobileNet-0.25 (The Latency-Accuracy Tradeoff)
To achieve our **~45ms inference latency** on a GTX 1650, we used the **MobileNet-0.25** backbone. It offers the best performance-per-watt, ensuring the system remains responsive even in dense multi-face scenarios without requiring expensive A100 GPUs.

### 4 · The Necessity of 5-Point Alignment
Verification accuracy drops by ~15-20% if faces are not aligned. Our pipeline performs an explicit **5-point landmark transformation** (eyes, nose, mouth corners) before generating embeddings, ensuring that the ArcFace backbone receives spatially consistent input.

---

## 🏗️ Project Structure

```text
Ratina_Face/
├── src/                    # Core logic — FaceDetector wrapper
├── tests/                  # Automated test suite (pytest)
├── .github/workflows/      # CI/CD pipelines (GitHub Actions)
├── app.py                  # Streamlit web application
├── main.py                 # Command-line tool
├── Dockerfile              # Production container setup
└── requirements.txt        # Managed dependencies
```

---

## 📦 Quick Start

### 1 · Clone & Install

```bash
git clone https://github.com/ImdataScientistSachin/RetinaFace-Detection
cd RetinaFace-Detection
python -m pip install -r requirements.txt
```

### 2 · Operational Monitoring (MLflow)

To view inference logs and latency benchmarks:

```bash
mlflow ui
```

### 3 · Deployment

```bash
docker build -t retinaface-pro .
docker run -p 7860:7860 retinaface-pro
```

---

## 👤 Author

**Sachin Paunikar** — [LinkedIn](https://www.linkedin.com/in/sachin-paunikar-datascientists) | [GitHub](https://github.com/ImdataScientistSachin)

<div align="center">
<sub>Built with ❤️ · Powered by RetinaFace & DeepFace · Deployed on HuggingFace Spaces</sub>
</div>
