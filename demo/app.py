from __future__ import annotations

import json
import os
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image

from gemini_service import EDIT_THEMES, api_key, caption_image, edit_image, make_edit_instruction

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
PROJECT_LINKS = [
    {"id": "paper", "label": "Paper", "title": "Unsafe2Safe paper", "url": "https://cvpr.thecvf.com/virtual/2026/poster/38097"},
    {"id": "project", "label": "Project page", "title": "Unsafe2Safe project page", "url": "https://see-ai-lab.github.io/unsafe2safe/"},
    {"id": "github", "label": "GitHub", "title": "Unsafe2Safe on GitHub", "url": "https://github.com/SEE-AI-Lab/unsafe2safe"},
    {"id": "dataset", "label": "Dataset", "title": "Unsafe2Safe dataset", "url": "https://huggingface.co/datasets/minhdinh2/Unsafe2Safe"},
    {"id": "minh", "label": "Minh Dinh Trong", "title": "Minh Dinh Trong’s page", "url": "https://trminh0711.github.io/"},
    {"id": "souyoung", "label": "SouYoung Jin", "title": "SouYoung Jin’s page", "url": "https://souyoungjin.github.io/"},
    {"id": "dartmouth", "label": "Dartmouth", "title": "Dartmouth College", "url": "https://home.dartmouth.edu/"},
]
PROJECT_LINKS_BY_ID = {link["id"]: link for link in PROJECT_LINKS}


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
    try:
        captions = caption_image(image)
    except Exception as exc:
        return jsonify(error=f"Gemini could not caption this photo: {exc}"), 502

    return jsonify(
        full_caption=captions["FULL_VISUAL_DESCRIPTION"],
        public_caption=captions["PRIVACY_PRESERVING_DESCRIPTION"],
        privacy_review=captions["PRIVACY_REVIEW"],
        privacy_changes=captions["PRIVACY_CHANGES"],
    )


@app.post("/edit-instruction")
def generate_edit_instruction():
    public_caption = request.form.get("public_caption", "").strip()
    mode = request.form.get("mode")
    privacy_review = json.loads(request.form.get("privacy_review", "[]"))
    if not public_caption:
        return jsonify(error="Generate Stage 1 captions before choosing an edit."), 400
    if mode not in EDIT_THEMES:
        return jsonify(error="Choose one of the three edit modes."), 400
    if not api_key():
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    try:
        instruction = make_edit_instruction(public_caption, mode, privacy_review)
    except Exception as exc:
        return jsonify(error=f"Gemini could not generate an edit instruction: {exc}"), 502
    if not instruction:
        return jsonify(error="Gemini returned no edit instruction. Try again."), 502
    return jsonify(edit_instruction=instruction)


@app.post("/edit")
def edit_photo():
    photo = request.files.get("photo")
    public_caption = request.form.get("public_caption", "").strip()
    edit_instruction = request.form.get("edit_instruction", "").strip()
    if photo is None:
        return jsonify(error="Choose a photo first."), 400
    if not public_caption:
        return jsonify(error="Generate Stage 1 captions before editing."), 400
    if not edit_instruction:
        return jsonify(error="Generate an edit instruction before editing."), 400
    if not photo.mimetype or not photo.mimetype.startswith("image/"):
        return jsonify(error="Please choose an image file."), 400
    if not api_key():
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    image = Image.open(BytesIO(photo.read())).convert("RGB")
    try:
        result = edit_image(image, public_caption, edit_instruction)
    except Exception as exc:
        return jsonify(error=f"Gemini could not edit this photo: {exc}"), 502
    if not result:
        return jsonify(error="Gemini returned no edited image. Try again with a different photo."), 502
    image_data, mime_type = result
    return jsonify(
        image=image_data,
        mime_type=mime_type,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), debug=False)
