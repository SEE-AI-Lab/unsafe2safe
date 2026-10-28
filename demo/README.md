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

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Choose **Upload a photo** or **Take a photo**; your browser asks for camera access the first time. Stage 1 inspects the image, generates private and public captions, and creates a themed edit instruction. Stage 2 uses Gemini to generate a safe image. Hoverable markers show privacy-review findings.

Stage 1 uses `gemini-3.5-flash-lite` for captions and edit-instruction generation. Stage 2 uses `gemini-3.1-flash-lite-image`. Set `GEMINI_CAPTION_MODEL` or `GEMINI_IMAGE_MODEL` to choose different models.

The demo prompts in `demo/prompts/` derive from the paper prompts on `main` (Appendix S6.1, Figures S9–S10). Gemini returns structured captions, image coordinates for privacy-review markers, and phrase pairs for highlights. The public caption generalizes sensitive attributes while preserving source facts and object presence; the generated edit instruction supplies concrete replacement appearances. The demo uses up to 5 review items instead of 10 and adds the selected theme and privacy-review findings to instruction generation. The image request chooses the supported aspect ratio closest to the source. The interactive Gemini flow is paper-inspired, not an exact reproduction of the paper's diffusion editor.
