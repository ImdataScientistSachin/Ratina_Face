import os
from huggingface_hub import upload_folder

"""
One-shot uploader for Hugging Face Spaces.
Usage: 
1. Set environment variable: export HF_TOKEN="your_token_here"
2. Run: python upload_to_hf.py
"""

# Configuration
REPO_ID = "ImdataScientistSachin/Ratina_face"
FOLDER_PATH = "."
HF_TOKEN = os.environ.get("HF_TOKEN")

def sync_to_hub():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable is not set.")
        print("Please set it using: set HF_TOKEN=your_token (Windows) or export HF_TOKEN=your_token (Linux/Mac)")
        return

    print(f"🚀 Starting sync to Hugging Face: {REPO_ID}")
    
    try:
        upload_folder(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
            ignore_patterns=[
                "examples/*",
                "__pycache__/*",
                "*.pyc",
                ".git/*",
                "upload_to_hf.py",
                ".github/*",
                "tests/*"
            ],
            commit_message="deploy: professional inference pipeline update"
        )
        print("✅ Sync complete!")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

if __name__ == "__main__":
    sync_to_hub()
