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

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). On mobile, tap the photo area to take a picture or choose one. Pick a Halloween privacy edit, broaden the cast, or move the moment to a new setting.

The default model is `gemini-3.1-flash-lite-image`. Set `GEMINI_IMAGE_MODEL` to use another model.
