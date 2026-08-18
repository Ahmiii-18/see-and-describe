# 👁️ See & Describe — AI Vision Assistant

An AI-powered web app that looks at an image (uploaded or from your camera) and describes what it sees in natural language, then reads the description out loud. Built as a first portfolio project for a BS Robotics & AI degree.

This mirrors a core building block of assistive robotics and scene-understanding systems: **perceive → understand → communicate.**

**🔗 Live demo:** https://see-and-describe-4fx7u5s5hldcnsxwabqdgn.streamlit.app/

## Screenshots

_Add a screenshot or short GIF here showing: image upload → generated caption → audio player._
`![demo](assets/demo.png)`

## Features

- **Image captioning** — describes what's in an uploaded photo or camera shot
- **Detailed mode** — an optional longer, more varied description (nucleus sampling) instead of BLIP's default short caption
- **Object detection** — optional YOLOv8 pass that draws bounding boxes and lists detected object classes
- **Multi-language output** — translate the caption and hear it spoken in Spanish, French, German, Urdu, Hindi, Arabic, Chinese, Japanese, or Turkish
- **Text-to-speech** — reads the (translated) caption aloud

## How it works

1. **Computer Vision (captioning)** — [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) (Salesforce), a pretrained vision-language model, analyzes the image and generates a caption.
2. **Computer Vision (detection)** — [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics), a pretrained object-detection model, optionally locates and labels objects in the image with bounding boxes.
3. **Translation** — [deep-translator](https://pypi.org/project/deep-translator/) (Google Translate backend) converts the English caption into the selected language.
4. **Text-to-Speech** — [gTTS](https://pypi.org/project/gTTS/) converts the caption into spoken audio. Note: both translation and speech require an internet connection — the app falls back gracefully to English text if either fails.
5. **Interface** — [Streamlit](https://streamlit.io) provides the web UI (upload, camera, options, results).

## Run locally

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd see-and-describe
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

> First run downloads the BLIP model weights (~1 GB), so it may take a few minutes depending on your connection. Subsequent runs are fast thanks to caching.

## Deploy for free (so you have a live link for LinkedIn)

### Option A: Hugging Face Spaces (recommended, built for ML apps)
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space** → choose **Streamlit** as the SDK
3. Upload `app.py`, `requirements.txt`, and this `README.md`
4. Space builds automatically and gives you a public URL like `https://huggingface.co/spaces/yourname/see-and-describe`

### Option B: Streamlit Community Cloud
1. Push this project to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Select the repo → deploy
4. You get a public `*.streamlit.app` link

## Possible extensions (good "v3" commits to show growth)
- Add a "describe surroundings continuously" mode using a live camera feed (robotics angle)
- Deploy the model on a Raspberry Pi + camera module for a physical assistive device
- Swap gTTS for a local/offline TTS engine (e.g. `pyttsx3`) for use without internet access
- Let the user choose which YOLO model size (nano/small/medium) to trade speed for accuracy
- Combine detected object labels into the caption prompt for more grounded descriptions
- Add distance/depth estimation for a stronger assistive-navigation angle

## Tech stack
`Python` `Streamlit` `Transformers (BLIP)` `PyTorch` `YOLOv8 (Ultralytics)` `deep-translator` `gTTS`

---
Built as a BSRAI portfolio project.
