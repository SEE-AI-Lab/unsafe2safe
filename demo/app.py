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
        "PRIVACY_REVIEW": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ITEM": {"type": "STRING"},
                    "REASON": {"type": "STRING"},
                    "X": {"type": "NUMBER"},
                    "Y": {"type": "NUMBER"},
                },
                "required": ["ITEM", "REASON", "X", "Y"],
            },
        },
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

EDIT_THEMES = {
    "spooky_private": "Give the privacy changes a tasteful Halloween character: a clearly different fictional face, hairstyle, unbranded costume, and subtle spooky atmosphere. No gore.",
    "diverse_cast": "Where people are changed, favor a natural, respectful mix of ages, skin tones, gender expressions, and visible abilities. Avoid stereotypes.",
    "new_setting": "If the background contains private location details, suggest an anonymous, visually distinct alternative that preserves the scene layout, object arrangement, and actions.",
}
EDIT_IDEA_PROMPT = (Path(__file__).resolve().parents[1] / "prompts/qwen_image_captioning-private-expand.txt").read_text(encoding="utf-8")


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
        "Return the requested paper sections using the supplied JSON schema. In PRIVACY_REVIEW, return up to "
        "5 objects with ITEM, REASON, and approximate X/Y coordinates as percentages from the top-left of the "
        "image (0 to 100). Place each point on the visible detail it describes. Also return PRIVACY_CHANGES "
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


@app.post("/edit-instruction")
def generate_edit_instruction():
    public_caption = request.form.get("public_caption", "").strip()
    mode = request.form.get("mode")
    if not public_caption:
        return jsonify(error="Generate Stage 1 captions before choosing an edit."), 400
    if mode not in EDIT_THEMES:
        return jsonify(error="Choose one of the three edit modes."), 400
    if not api_key():
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    prompt = (
        f"{EDIT_IDEA_PROMPT.format(public_caption=public_caption)}\n\n"
        f"Selected theme: {EDIT_THEMES[mode]}\n"
        "Apply the theme only to the proposed identity/privacy modifications. Follow all rules in the prompt, "
        "especially preserving scene layout and actions. Return only the generated edit instruction."
    )
    client = genai.Client(api_key=api_key())
    try:
        response = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=prompt,
            config={"max_output_tokens": 160},
        )
    except Exception as exc:
        return jsonify(error=f"Gemini could not generate an edit instruction: {exc}"), 502
    instruction = (response.text or "").strip()
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
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=92)
    prompt = (
        "Edit the supplied photo to match the privacy-preserving scene description and follow the selected "
        "edit instruction visibly. Treat the public description as the privacy boundary: do not restore details "
        "it omits. Preserve the original framing and useful non-sensitive content unless the edit instruction "
        "asks to change them. Preserve the original aspect ratio, camera framing, and spatial arrangement. "
        "Make requested changes clear in the image, keep the result photorealistic, and do not add identifying "
        "personal details.\n"
        f"Public description: {public_caption}\nEdit instruction: {edit_instruction}"
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
