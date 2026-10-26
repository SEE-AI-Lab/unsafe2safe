# SafeFrame photo demo

A small local web app for uploading or taking a photo, requesting a privacy edit, and downloading the result. The Gemini key stays on the server.

From the repository root:

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
export GEMINI_API_KEY="your-key"
python demo/app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) on a computer or phone on the same device. On mobile, tap the photo area to use the camera or choose an existing image. The default editor is `gemini-3.1-flash-lite-image`; change it with `GEMINI_IMAGE_MODEL` if needed.
