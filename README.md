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

**RetinaFace Pro** is a deployed, production-ready computer vision pipeline built on the state-of-the-art **RetinaFace** architecture. It refactors cutting-edge research into a fully usable suite that supports real-time multi-face detection, precise landmark extraction, and identity verification via **DeepFace (ArcFace)** embeddings.

The model simultaneously predicts:

| Capability | Description |
|---|---|
| 🔲 **Face Detection** | Highly accurate bounding boxes across multiple faces |
| 📍 **5-Point Landmarks** | High-precision localisation — eyes, nose, and mouth corners |
| 🪪 **Identity Verification** | One-to-many matching with ArcFace embeddings |
| 📊 **Confidence Scores** | Per-detection quality measurements |

---

## 🌟 Key Features

- **Multi-Face Analysis** — Detect and extract portraits from group shots in a single inference pass
- **1-to-Many Identity Check** — Compare a reference persona against multiple test images with detailed distance metrics
- **Dual Interface** — Choose between an interactive web dashboard or a powerful CLI for batch processing
- **Production Standard** — Fully Dockerized and pre-configured for free deployment on Hugging Face Spaces

---

## 🧬 Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | RetinaFace (SOTA backbone) |
| **Verification** | DeepFace · ArcFace |
| **Backend** | Python 3.10+ · TensorFlow · OpenCV |
| **Web UI** | Streamlit |
| **Deployment** | Docker · HuggingFace Spaces |

---

## 🏗️ Project Structure

```text
Ratina_Face/
├── src/                    # Core logic — FaceDetector wrapper
├── examples/               # Sample images & test data
├── tests/                  # Automated test suite (pytest)
├── .github/workflows/      # CI/CD pipelines (GitHub Actions)
├── app.py                  # Streamlit web application
├── main.py                 # Command-line tool
├── Dockerfile              # Production container setup
└── requirements.txt        # Managed dependencies
```

---

## 🚀 Quick Start

### 1 · Clone & Install

```bash
git clone https://github.com/ImdataScientistSachin/Ratina-Face
cd Ratina-Face
python -m pip install -r requirements.txt
```

> [!TIP]
> On Windows, always use the `python -m` prefix to avoid path-space errors.

### 2 · Launch Web Dashboard

```bash
python -m streamlit run app.py
```

### 3 · Command-Line Usage

```bash
# Detect and process a folder of images
python main.py detect --input_dir examples/ --output_dir results/

# Verify identity match between two images
python main.py verify examples/user1.jpg examples/user2.jpg
```

---

## 🐳 Docker & Deployment

Deploy to Hugging Face Spaces or any cloud provider using the optimised Docker setup:

```bash
docker build -t retinaface-pro .
docker run -p 7860:7860 retinaface-pro
```

> Continuous integration is handled automatically via **GitHub Actions** — linting and pytest run on every push.

---

## 📄 License

Distributed under the **MIT License** — free to use, modify, and distribute.
See [`LICENSE`](./LICENSE) for details.

---

## 👤 Author

**Sachin Paunikar**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sachin-paunikar-datascientists)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/ImdataScientistSachin)

---

<div align="center">
<sub>Built with ❤️ · Powered by RetinaFace & DeepFace · Deployed on HuggingFace Spaces</sub>
</div>
