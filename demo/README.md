# SafeFrame photo demo

A small local app for taking or uploading a photo, choosing one of three edits, and downloading the Gemini result. The API key stays on the server.

From the repository root:

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
export GEMINI_API_KEY="your-key"
python demo/app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Choose **Upload a photo** or **Take a photo**; your browser asks for camera access the first time. Stage 1 shows the full and privacy-safe captions, with hoverable markers for privacy-review findings. Stage 2 runs the paper's edit-idea prompt through Gemini; the three presets provide theme cues to that generated instruction.

Stage 1 defaults to `gemini-3.5-flash-lite`; Stage 2 defaults to `gemini-3.1-flash-lite-image`. Set `GEMINI_CAPTION_MODEL` or `GEMINI_IMAGE_MODEL` to choose different models.
