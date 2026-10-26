from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
from google import genai

app = Flask(__name__)
CAPTION_MODEL = os.environ.get("GEMINI_CAPTION_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")
STAGE1_PROMPT = (Path(__file__).resolve().parents[1] / "prompts/intern_image_captioning.txt").read_text(encoding="utf-8")
STAGE1_SYSTEM = "You are a vision-language assistant that generates privacy-aware, structured descriptions of real-world images."
STAGE1_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "PRIVACY_REVIEW": {"type": "STRING"},
        "FULL_VISUAL_DESCRIPTION": {"type": "STRING"},
        "PRIVACY_PRESERVING_DESCRIPTION": {"type": "STRING"},
        "PRIVACY_CHANGES": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "PRIVATE_PHRASE": {"type": "STRING"},
                    "PUBLIC_PHRASE": {"type": "STRING"},
                },
                "required": ["PRIVATE_PHRASE", "PUBLIC_PHRASE"],
            },
        },
    },
    "required": ["PRIVACY_REVIEW", "FULL_VISUAL_DESCRIPTION", "PRIVACY_PRESERVING_DESCRIPTION", "PRIVACY_CHANGES"],
}
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
        "Turn the person into a tasteful Halloween character and change their face enough that they are "
        "no longer recognizable as the same person. Change the face shape and visible features, including "
        "the eyes, nose, and jaw; do not just cover the unchanged face with a mask. Add fitting makeup, "
        "a mask, or costume details. Preserve the pose, framing, setting, and useful non-identifying content. "
        "Keep it photorealistic and avoid gore."
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
    prompt = (
        f"{STAGE1_SYSTEM}\n\n{STAGE1_PROMPT.format(image_class='photo')}\n\n"
        "Return the requested paper sections using the supplied JSON schema. Also return PRIVACY_CHANGES "
        "as pairs of exact phrases: PRIVATE_PHRASE must appear in FULL_VISUAL_DESCRIPTION, and "
        "PUBLIC_PHRASE must appear in PRIVACY_PRESERVING_DESCRIPTION. Use an empty PUBLIC_PHRASE "
        "when the detail was omitted. Order pairs by appearance; return an empty array if nothing changed."
    )
    client = genai.Client(api_key=api_key())
    try:
        response = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=[prompt, image],
            config={
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "response_schema": STAGE1_RESPONSE_SCHEMA,
            },
        )
    except Exception as exc:
        return jsonify(error=f"Gemini could not caption this photo: {exc}"), 502

    captions = json.loads(response.text)
    return jsonify(
        full_caption=captions["FULL_VISUAL_DESCRIPTION"],
        public_caption=captions["PRIVACY_PRESERVING_DESCRIPTION"],
        privacy_review=captions["PRIVACY_REVIEW"],
        privacy_changes=captions["PRIVACY_CHANGES"],
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
