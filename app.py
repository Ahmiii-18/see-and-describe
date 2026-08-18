"""
See & Describe — AI Image Captioning + Voice Assistant
BSRAI Project — Computer Vision (BLIP + YOLOv8) + Text-to-Speech + Translation.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io

import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from deep_translator import GoogleTranslator

st.set_page_config(page_title="See & Describe", page_icon="🤖", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Language name -> code (shared between gTTS speech and Google Translate)
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Turkish": "tr",
}


# --------------------------------------------------------------------------
# Model loading (cached — runs once per session)
# --------------------------------------------------------------------------

@st.cache_resource
def load_caption_model():
    """Load the BLIP image captioning model once and cache it for the session."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_detection_model():
    """Load the YOLOv8 nano object-detection model once and cache it.
    Returns None if ultralytics isn't installed or the weights can't be fetched,
    so object detection can be gracefully disabled instead of crashing the app."""
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------

def generate_caption(image: Image.Image, processor, model, detailed: bool = False) -> str:
    """Run the image through BLIP and return a natural-language caption.

    detailed=True uses nucleus sampling and a higher token budget for a
    longer, more varied description instead of BLIP's default short caption.
    """
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if detailed:
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            )
        else:
            out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def detect_objects(image: Image.Image, yolo_model):
    """Run YOLOv8 on the image. Returns (annotated_image, sorted list of
    unique object labels detected), or (None, []) if detection fails."""
    try:
        results = yolo_model.predict(image, verbose=False)
        annotated_bgr = results[0].plot()  # numpy array, BGR channel order
        annotated_rgb = Image.fromarray(annotated_bgr[:, :, ::-1])

        labels = set()
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            labels.add(results[0].names[class_id])

        return annotated_rgb, sorted(labels)
    except Exception as e:
        st.warning(f"Object detection failed: {e}")
        return None, []


def translate_caption(text: str, target_lang_code: str) -> str:
    """Translate the English caption into the target language.
    Falls back to the original English text if translation fails
    (e.g. no internet connection)."""
    if target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang_code).translate(text)
    except Exception as e:
        st.warning(f"Couldn't translate (showing English instead): {e}")
        return text


def text_to_speech(text: str, lang_code: str = "en") -> io.BytesIO | None:
    """Convert text to spoken audio in the given language. Returns None if
    TTS fails (e.g. no internet connection, since gTTS requires network access)."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.warning(f"Couldn't generate audio (caption still works fine): {e}")
        return None


def show_image(image, caption: str):
    """Display an image at full column width, regardless of which Streamlit
    version is installed (the parameter name for this has changed across
    versions: use_column_width -> use_container_width -> width='stretch')."""
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image, caption=caption)


def load_image_safely(file) -> Image.Image | None:
    """Open an uploaded/camera file as RGB, handling corrupt or unsupported files."""
    try:
        return Image.open(file).convert("RGB")
    except Exception as e:
        st.error(f"Couldn't read that image: {e}")
        return None


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main():
    st.title("🤖 AI Automation (See & Describe)")
    st.caption(
        "An AI vision assistant that looks at an image and describes it in "
        "natural language — the same core idea behind assistive robotics "
        "and scene-understanding systems."
    )

    try:
        with st.spinner("Loading AI model (first run may take a minute)..."):
            processor, model = load_caption_model()
    except Exception as e:
        st.error(f"Failed to load the captioning model: {e}")
        st.stop()

    st.divider()

    source = st.radio("Choose image source:", ["Upload a photo", "Use camera"])

    image = None
    if source == "Upload a photo":
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = load_image_safely(uploaded_file)
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = load_image_safely(camera_image)

    if image is not None:
        show_image(image, "Your image")

        with st.expander("⚙️ Options", expanded=False):
            detailed = st.checkbox(
                "📝 Detailed description (longer, more varied caption)"
            )
            detect_objs = st.checkbox(
                "🎯 Detect objects (YOLOv8, draws bounding boxes)"
            )
            language_name = st.selectbox(
                "🔊 Speech / caption language", list(LANGUAGES.keys())
            )
            lang_code = LANGUAGES[language_name]

        if st.button("🔍 Describe this image", type="primary"):
            # --- Caption ---
            try:
                with st.spinner("Analyzing image..."):
                    caption_en = generate_caption(image, processor, model, detailed)
            except Exception as e:
                st.error(f"Couldn't analyze this image: {e}")
                return

            display_caption = caption_en
            if lang_code != "en":
                with st.spinner(f"Translating to {language_name}..."):
                    display_caption = translate_caption(caption_en, lang_code)

            st.success("Here's what I see:")
            st.markdown(f"### 🗣️ \"{display_caption}\"")
            if lang_code != "en":
                st.caption(f"English: \"{caption_en}\"")

            # --- Object detection ---
            if detect_objs:
                yolo_model = load_detection_model()
                if yolo_model is None:
                    st.info(
                        "Object detection isn't available — make sure "
                        "`ultralytics` is installed (see requirements.txt)."
                    )
                else:
                    with st.spinner("Detecting objects..."):
                        annotated_image, labels = detect_objects(image, yolo_model)
                    if annotated_image is not None:
                        show_image(annotated_image, "Detected objects")
                        if labels:
                            st.markdown(f"**Objects found:** {', '.join(labels)}")
                        else:
                            st.caption("No objects confidently detected.")

            # --- Audio ---
            with st.spinner("Generating audio..."):
                audio = text_to_speech(display_caption, lang_code)
            if audio is not None:
                st.audio(audio, format="audio/mp3")

    st.divider()
    st.caption(
        "Built with BLIP (Salesforce) for captioning, YOLOv8 (Ultralytics) for "
        "object detection, Google Translate for language support, and gTTS for "
        "speech. Part of a BSRAI portfolio project — see the GitHub repo for details."
    )


if __name__ == "__main__":
    main()
