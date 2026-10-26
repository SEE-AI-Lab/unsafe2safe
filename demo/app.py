from __future__ import annotations

import base64
import os
from io import BytesIO

from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from google import genai


app = Flask(__name__)
MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")

PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f7f7f3">
  <title>SafeFrame — private photo edits</title>
  <style>
    :root { color-scheme: light; --ink:#20231f; --muted:#737970; --line:#dedfd8; --paper:#f7f7f3; --white:#fff; --green:#325c48; --green2:#e9f0e9; }
    * { box-sizing:border-box }
    body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif }
    header { height:68px; display:flex; align-items:center; justify-content:space-between; padding:0 max(24px,calc((100vw - 1080px)/2)); border-bottom:1px solid var(--line); background:#fbfbf8 }
    .brand { display:flex; align-items:center; gap:10px; font-weight:680; letter-spacing:-.03em }
    .mark { width:29px;height:29px;border-radius:9px;background:var(--green);display:grid;place-items:center;color:white;font-size:17px }
    .tag { color:var(--muted); font-size:13px }
    main { max-width:1080px; margin:0 auto; padding:60px 24px 80px }
    .intro { max-width:570px; margin:0 0 34px }
    .eyebrow { text-transform:uppercase; letter-spacing:.12em; color:var(--green); font-size:11px; font-weight:700 }
    h1 { font-size:clamp(34px,5vw,52px); letter-spacing:-.055em; line-height:1.04; margin:13px 0 14px; font-weight:620 }
    .intro p { color:var(--muted); margin:0; font-size:16px }
    .workspace { display:grid; grid-template-columns:minmax(280px,.82fr) minmax(360px,1.18fr); gap:22px; align-items:stretch }
    .panel { background:var(--white); border:1px solid var(--line); border-radius:18px; padding:22px }
    .panel-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:17px }
    .panel-title { font-weight:650; font-size:15px }
    .step { font-size:11px; color:var(--muted); border:1px solid var(--line); padding:4px 8px; border-radius:20px }
    .drop { display:grid; place-items:center; min-height:240px; border:1px dashed #bfc5bc; background:#fafbf8; border-radius:13px; overflow:hidden; position:relative; cursor:pointer }
    .drop:hover { border-color:var(--green); background:#f5f8f3 }
    .drop img { width:100%; height:100%; max-height:360px; object-fit:contain; display:none }
    .drop.has-image img { display:block }
    .empty { text-align:center; padding:28px; color:var(--muted) }
    .upload-icon { width:43px;height:43px;border-radius:13px;background:var(--green2);color:var(--green);display:grid;place-items:center;margin:0 auto 12px;font-size:22px }
    .empty strong { color:var(--ink); display:block; font-size:14px; margin-bottom:3px }
    .empty small { font-size:12px }
    #photo { position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer }
    label.prompt-label { display:block;font-size:13px;font-weight:620;margin:18px 0 7px }
    textarea { display:block;resize:vertical;width:100%;min-height:80px;padding:11px 12px;border:1px solid var(--line);border-radius:10px;font:inherit;font-size:14px;line-height:1.5;color:var(--ink);background:white }
    textarea:focus { outline:2px solid #b9cbbd; border-color:var(--green) }
    .hint { font-size:11px;color:var(--muted);margin:7px 0 0 }
    button { margin-top:17px;width:100%;border:0;border-radius:10px;background:var(--green);color:white;font:inherit;font-size:14px;font-weight:600;padding:13px 16px;cursor:pointer;transition:background .15s,transform .15s }
    button:hover { background:#274b39;transform:translateY(-1px) }
    button:disabled { opacity:.55;cursor:wait;transform:none }
    .result { min-height:100%;display:flex;flex-direction:column }
    .result-stage { flex:1;display:grid;place-items:center;min-height:340px;border-radius:13px;background:#f5f6f2;overflow:hidden }
    .placeholder { max-width:250px;padding:25px;text-align:center;color:var(--muted);font-size:13px }
    .placeholder .spark { font-size:26px;color:#819781;margin-bottom:8px }
    #resultImage { display:none;width:100%;max-height:520px;object-fit:contain }
    #download { display:none;text-align:center;margin-top:14px;color:var(--green);font-size:13px;font-weight:650;text-decoration:none }
    #status { min-height:20px;margin:12px 0 0;color:var(--muted);font-size:12px }
    #status.error { color:#a23e34 }
    .note { margin-top:18px;display:flex;gap:9px;color:var(--muted);font-size:11px;line-height:1.45 }
    .note span { color:var(--green) }
    @media (max-width:760px) { header { height:60px } main { padding:38px 18px 55px } .workspace { grid-template-columns:1fr } .result-stage { min-height:250px } .intro { margin-bottom:24px } }
  </style>
</head>
<body>
  <header><div class="brand"><span class="mark">s</span> SafeFrame</div><div class="tag">A more private picture</div></header>
  <main>
    <section class="intro"><div class="eyebrow">Private photo editing</div><h1>Keep the moment.<br>Change what’s personal.</h1><p>Choose a photo and tell us what you’d like changed. Your edited image will appear here.</p></section>
    <section class="workspace">
      <div class="panel">
        <div class="panel-head"><div class="panel-title">Your photo</div><div class="step">01 / ADD</div></div>
        <label class="drop" id="drop" for="photo"><div class="empty" id="empty"><div class="upload-icon">↑</div><strong>Choose or take a photo</strong><small>JPG or PNG · camera opens on mobile</small></div><img id="preview" alt="Selected photo"><input id="photo" type="file" accept="image/*" capture="environment"></label>
        <label class="prompt-label" for="instruction">What should change?</label>
        <textarea id="instruction" placeholder="For example: remove the house number from the background">Remove identifying details while keeping the people, objects, and overall scene unchanged.</textarea>
        <p class="hint">Describe only the change. The rest of the photo should stay as close as possible.</p>
        <button id="edit" disabled>Make a private copy</button>
        <div class="note"><span>●</span><div>Your photo is sent to Gemini to create the edit. It isn’t saved by this demo.</div></div>
      </div>
      <div class="panel result">
        <div class="panel-head"><div class="panel-title">Edited photo</div><div class="step">02 / RESULT</div></div>
        <div class="result-stage"><div class="placeholder" id="placeholder"><div class="spark">✳</div>Your edited photo will show up here.</div><img id="resultImage" alt="Gemini edited result"></div>
        <a id="download" download="safeframe-edited.png">Download edited photo ↓</a>
        <div id="status" role="status" aria-live="polite"></div>
      </div>
    </section>
  </main>
  <script>
    const input = document.getElementById('photo');
    const preview = document.getElementById('preview');
    const edit = document.getElementById('edit');
    const status = document.getElementById('status');
    let selectedFile;
    input.addEventListener('change', () => {
      selectedFile = input.files[0];
      if (!selectedFile) return;
      preview.src = URL.createObjectURL(selectedFile);
      document.getElementById('drop').classList.add('has-image');
      document.getElementById('empty').style.display = 'none';
      edit.disabled = false;
      document.getElementById('resultImage').style.display = 'none';
      document.getElementById('placeholder').style.display = 'block';
      document.getElementById('download').style.display = 'none';
      status.textContent = '';
      status.className = '';
    });
    edit.addEventListener('click', async () => {
      if (!selectedFile) return;
      edit.disabled = true;
      edit.textContent = 'Editing your photo…';
      status.textContent = 'This can take a few seconds.';
      status.className = '';
      const form = new FormData();
      form.append('photo', selectedFile);
      form.append('instruction', document.getElementById('instruction').value);
      try {
        const response = await fetch('/edit', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'The edit could not be completed.');
        const url = `data:${data.mime_type};base64,${data.image}`;
        document.getElementById('resultImage').src = url;
        document.getElementById('resultImage').style.display = 'block';
        document.getElementById('placeholder').style.display = 'none';
        const download = document.getElementById('download');
        download.href = url;
        download.style.display = 'block';
        status.textContent = 'Your edited photo is ready.';
      } catch (error) {
        status.textContent = error.message;
        status.className = 'error';
      } finally {
        edit.disabled = false;
        edit.textContent = 'Make a private copy';
      }
    });
  </script>
</body>
</html>"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.post("/edit")
def edit_photo():
    photo = request.files.get("photo")
    if photo is None:
        return jsonify(error="Choose a photo first."), 400
    if not photo.mimetype or not photo.mimetype.startswith("image/"):
        return jsonify(error="Please choose an image file."), 400
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return jsonify(error="Set GEMINI_API_KEY in the terminal, then restart the app."), 503

    image_bytes = photo.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=92)
    prompt = (
        "Edit this photo to remove or change identifying personal details while preserving the people, "
        "objects, composition, and natural photographic style. Apply this specific request: "
        f"{request.form.get('instruction', '').strip()}"
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.interactions.create(
            model=MODEL,
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": base64.b64encode(image_buffer.getvalue()).decode("ascii"), "mime_type": "image/jpeg"},
            ],
            response_format={"type": "image", "image_size": "1K"},
        )
    except Exception as exc:
        return jsonify(error=f"Gemini could not edit this photo: {exc}"), 502
    if not response.output_image:
        return jsonify(error="Gemini returned no edited image. Try another photo or instruction."), 502
    return jsonify(image=response.output_image.data, mime_type=getattr(response.output_image, "mime_type", None) or "image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), debug=False)
