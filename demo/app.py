from __future__ import annotations

import base64
import os
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image
from google import genai


app = Flask(__name__)
MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")

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


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/edit")
def edit_photo():
    photo = request.files.get("photo")
    mode = request.form.get("mode")
    if photo is None:
        return jsonify(error="Choose a photo first."), 400
    if mode not in PRESETS:
        return jsonify(error="Choose one of the three edit modes."), 400
    if not photo.mimetype or not photo.mimetype.startswith("image/"):
        return jsonify(error="Please choose an image file."), 400

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    image = Image.open(BytesIO(photo.read())).convert("RGB")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=92)
    prompt = (
        "Edit the supplied photo according to this selected mode. Preserve its useful content and "
        "photographic style except where the mode asks for a change.\n\n" + PRESETS[mode]
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.interactions.create(
            model=MODEL,
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
