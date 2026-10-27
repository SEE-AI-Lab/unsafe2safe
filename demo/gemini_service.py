"""Gemini prompts and model calls used by the demo routes."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path

from google import genai

CAPTION_MODEL = os.environ.get("GEMINI_CAPTION_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")
STAGE1_PROMPT = (Path(__file__).resolve().parent / "prompts/caption.txt").read_text(encoding="utf-8")
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
EDIT_THEMES = {
    "spooky_private": "Give the privacy changes a tasteful Halloween character: a clearly different fictional face, hairstyle, unbranded costume, and subtle spooky atmosphere. No gore.",
    "diverse_cast": "Where people are changed, favor a natural, respectful mix of ages, skin tones, gender expressions, and visible abilities. Avoid stereotypes.",
    "new_setting": "Relocate the same kind of scene to a different region, keeping its activity, composition, scene layout, object arrangement, and camera framing. Replace every private or identifying detail with a plausible generic alternative; do not show identifiable landmarks, names, or signage.",
}
EDIT_IDEA_PROMPT = (Path(__file__).resolve().parents[1] / "prompts/qwen_image_captioning-private-expand.txt").read_text(encoding="utf-8")


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def caption_image(image):
    prompt = (
        f"{STAGE1_SYSTEM}\n\n{STAGE1_PROMPT}\n\n"
        "Return the requested paper sections using the supplied JSON schema. In PRIVACY_REVIEW, return up to "
        "5 objects with ITEM, REASON, and approximate X/Y coordinates as percentages from the top-left of the "
        "image (0 to 100). Place each point on the visible detail it describes. Also return PRIVACY_CHANGES "
        "as pairs of exact phrases: PRIVATE_PHRASE must appear in FULL_VISUAL_DESCRIPTION, and "
        "PUBLIC_PHRASE must appear in PRIVACY_PRESERVING_DESCRIPTION. Use an empty PUBLIC_PHRASE "
        "when the detail was omitted. Order pairs by appearance; return an empty array if nothing changed."
    )
    with genai.Client(api_key=api_key()) as client:
        response = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=[prompt, image],
            config={
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "response_schema": STAGE1_RESPONSE_SCHEMA,
            },
        )
    return json.loads(response.text)


def make_edit_instruction(public_caption, mode, privacy_review):
    prompt = (
        f"{EDIT_IDEA_PROMPT.format(public_caption=public_caption)}\n\n"
        f"Selected theme: {EDIT_THEMES[mode]}\n"
        f"Stage 1 privacy review: {json.dumps(privacy_review, ensure_ascii=False)}\n"
        "Apply the theme only to the proposed identity/privacy modifications. Follow all rules in the prompt, "
        "especially preserving scene layout and actions. Explicitly account for every Stage 1 privacy-review item: "
        "remove or replace each flagged identity, attribute, text, logo, or location detail in the generated edit "
        "instruction, while keeping the public caption as the privacy boundary. Do not repeat private names or "
        "identifying text. Return only the generated edit instruction."
    )
    with genai.Client(api_key=api_key()) as client:
        response = client.models.generate_content(
            model=CAPTION_MODEL,
            contents=prompt,
            config={"max_output_tokens": 160},
        )
    return (response.text or "").strip()


def edit_image(image, public_caption, edit_instruction):
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=92)
    prompt = (
        "Edit the supplied photo to match the privacy-preserving scene description and follow the selected "
        "edit instruction visibly. Treat the public description as the privacy boundary: do not restore details "
        "it omits. Preserve the original framing and useful non-sensitive content unless the edit instruction "
        "asks to change them. Preserve the original aspect ratio, camera framing, and spatial arrangement, even "
        "when relocating the scene to another region. "
        "Make requested changes clear in the image, keep the result photorealistic, and do not add identifying "
        "personal details.\n"
        f"Public description: {public_caption}\nEdit instruction: {edit_instruction}"
    )
    with genai.Client(api_key=api_key()) as client:
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
    if not response.output_image:
        return None
    return response.output_image.data, getattr(response.output_image, "mime_type", None) or "image/png"
