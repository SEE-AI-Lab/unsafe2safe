from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request
from PIL import Image
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage1.output_parser import parse_structured_output


app = Flask(__name__)
CAPTION_MODEL = os.environ.get("GEMINI_CAPTION_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")
STAGE1_PROMPT = (Path(__file__).resolve().parents[1] / "prompts/intern_image_captioning.txt").read_text(encoding="utf-8")
STAGE1_SYSTEM = "You are a vision-language assistant that generates privacy-aware, structured descriptions of real-world images."
PROJECT_LINKS = [
    {"id": "paper", "label": "Paper", "title": "Unsafe2Safe paper", "url": "https://arxiv.org/abs/2603.28605"},
    {"id": "project", "label": "Project page", "title": "Unsafe2Safe project page", "url": "https://see-ai-lab.github.io/unsafe2safe/"},
    {"id": "github", "label": "GitHub", "title": "Unsafe2Safe on GitHub", "url": "https://github.com/SEE-AI-Lab/unsafe2safe"},
    {"id": "dataset", "label": "Dataset", "title": "Unsafe2Safe dataset", "url": "https://huggingface.co/datasets/minhdinh2/Unsafe2Safe"},
    {"id": "minh", "label": "Minh Dinh Trong", "title": "Minh Dinh Trong’s page", "url": "https://trminh0711.github.io/"},
    {"id": "souyoung", "label": "SouYoung Jin", "title": "SouYoung Jin’s page", "url": "https://souyoungjin.github.io/"},
    {"id": "dartmouth", "label": "Dartmouth", "title": "Dartmouth College", "url": "https://home.dartmouth.edu/"},
]
PROJECT_LINKS_BY_ID = {link["id"]: link for link in PROJECT_LINKS}

PRESETS = {
    "spooky_private": (
        "Give this photo a tasteful Halloween atmosphere while hiding identifying details. "
        "Use natural costume details, masks, or soft shadows to obscure recognizable features. "
        "Keep the scene and non-identifying content intact; avoid horror or gore."
    ),
    "diverse_cast": (
        "Reimagine the people in this photo as a group with as much variety as possible in age, "
        "skin tone, gender expression, and visible abilities. Keep their activity and setting intact. "
        "Make the result natural and respectful, without stereotypes."
    ),
    "new_setting": (
        "Keep the people and the moment in this photo, but move them to an unexpected, vivid new setting. "
        "Make the new place feel believable and preserve the people’s pose and activity."
    ),
}


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@app.get("/")
def home():
    return render_template("index.html", project_links=PROJECT_LINKS, links_by_id=PROJECT_LINKS_BY_ID)


@app.post("/caption")
def caption_photo():
    photo = request.files.get("photo")
    if photo is None or not photo.mimetype or not photo.mimetype.startswith("image/"):
        return jsonify(error="Choose an image file."), 400
    if not api_key():
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    image = Image.open(BytesIO(photo.read())).convert("RGB")
    prompt = f"{STAGE1_SYSTEM}\n\n{STAGE1_PROMPT.format(image_class='photo')}"
    client = genai.Client(api_key=api_key())
    try:
        response = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=[prompt, image],
            config={"max_output_tokens": 1024},
        )
    except Exception as exc:
        return jsonify(error=f"Gemini could not caption this photo: {exc}"), 502

    captions = parse_structured_output(response.text or "")
    return jsonify(
        full_caption=captions.get("PRIVATE_CAPTION", ""),
        public_caption=captions.get("PUBLIC_CAPTION", ""),
        privacy_review=captions.get("PRIVACY_REVIEW", ""),
    )


@app.post("/edit")
def edit_photo():
    photo = request.files.get("photo")
    mode = request.form.get("mode")
    public_caption = request.form.get("public_caption", "").strip()
    if photo is None:
        return jsonify(error="Choose a photo first."), 400
    if mode not in PRESETS:
        return jsonify(error="Choose one of the three edit modes."), 400
    if not public_caption:
        return jsonify(error="Generate Stage 1 captions before editing."), 400
    if not photo.mimetype or not photo.mimetype.startswith("image/"):
        return jsonify(error="Please choose an image file."), 400
    if not api_key():
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    image = Image.open(BytesIO(photo.read())).convert("RGB")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=92)
    prompt = (
        "Edit the supplied photo to match this privacy-preserving scene description and edit instruction. "
        "Preserve the original framing, useful non-sensitive content, and photorealistic style. "
        "Do not add identifying personal details.\n"
        f"Public description: {public_caption}\nEdit instruction: {PRESETS[mode]}"
    )
    client = genai.Client(api_key=api_key())
    try:
        response = client.interactions.create(
            model=IMAGE_MODEL,
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
                    "mime_type": "image/jpeg",
                },
            ],
            response_format={"type": "image", "image_size": "1K"},
        )
    except Exception as exc:
        return jsonify(error=f"Gemini could not edit this photo: {exc}"), 502
    if not response.output_image:
        return jsonify(error="Gemini returned no edited image. Try again with a different photo."), 502
    return jsonify(
        image=response.output_image.data,
        mime_type=getattr(response.output_image, "mime_type", None) or "image/png",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), debug=False)
