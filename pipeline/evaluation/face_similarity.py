import os

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def load_image(path, size=(512, 512)):
    with Image.open(path) as image:
        return np.array(image.convert("RGB").resize(size))


def get_embeddings(app, image):
    """Return all detected face embeddings in an image."""
    return [face.embedding for face in app.get(image)]


def _load_app():
    app = FaceAnalysis(name="antelopev2/antelopev2", root="models/insightface", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def nearest_face_similarity(image_root, synthetic_root, items):
    """Compute the paper's FaceSim using each original face's nearest match."""
    app = _load_app()

    similarities = []
    for item in items:
        original_faces = get_embeddings(app, load_image(os.path.join(image_root, item)))
        synthetic_faces = get_embeddings(app, load_image(os.path.join(synthetic_root, item)))
        for original in original_faces:
            if synthetic_faces:
                similarities.append(max(cosine_similarity([original], [synthetic])[0][0] for synthetic in synthetic_faces))
    return float(np.mean(similarities)) if similarities else float("nan")
