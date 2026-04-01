# MLOps Strategy: RetinaFace Pro

This project implements a **Modern MLOps Lifecycle** for computer vision. Use this document to explain your engineering decisions to technical recruiters.

---

## 1. Reproducibility (Docker)
*   **Decision:** Why use Docker instead of a simple conda environment?
*   **Explanation:** "I used a multi-stage Docker build based on `python:3.10-slim`. This ensures that my OpenCV and TensorFlow-Lite dependencies are locked to specific system libraries (`libgl1`, `libglib2.0-0`). This eliminates the 'it works on my machine' problem and allows the pipeline to scale horizontally across any cloud provider."

## 2. Continuous Delivery (CI/CD)
*   **Workflow:** GitHub Actions + Hugging Face Hub.
*   **Explanation:** "I implemented a custom GitHub Action workflow that performs two critical tasks: 
    1. **Automated Testing:** Every push runs `pytest` to verify the `FaceDetector` wrapper logic.
    2. **Python-Native Sync:** Instead of standard Git-sync (which struggles with large vision binaries), I used the `huggingface_hub` Python library to push a clean, history-free snapshot directly to the HF Space. This creates an immutable deployment record."

## 3. Observability & Monitoring (MLflow)
*   **Metric Tracking:** Inference Latency & Confidence Drift.
*   **Explanation:** "For production observability, I instrumented the inference pipeline with **MLflow**. While it's a pre-trained backbone, tracking metrics is still vital. I log:
    - **Inference Latency:** To detect CPU/GPU resource contention.
    - **Confidence Scores:** To monitor for 'Data Drift' (e.g., if image quality from the source hardware degrades over time)."

## 4. Model Selection Logic
*   **The Problem:** Why not just use HAAR cascades or MTCNN?
*   **The Answer:** "RetinaFace is a **single-stage pixel-wise detector** that handles tiny faces and extreme orientations far better than HAAR. By combining it with **ArcFace (99.8% LFW accuracy)**, I achieved a system that is robust enough for enterprise-grade 1-to-Many identity verification."
