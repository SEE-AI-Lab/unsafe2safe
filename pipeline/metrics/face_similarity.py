import os
from itertools import combinations

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def load_image(path, size=(512, 512)):
    with Image.open(path) as image:
        return np.array(image.convert("RGB").resize(size))


def get_embedding(app, image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].embedding


def compare_id_consistency(image_root, synthetic_root, items):
    """Compare face-embedding similarity between original and synthetic images."""
    app = FaceAnalysis(name="antelopev2/antelopev2", root="models/insightface", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))

    original_embeddings = []
    synthetic_embeddings = []
    for item in items:
        original = get_embedding(app, load_image(os.path.join(image_root, item)))
        synthetic = get_embedding(app, load_image(os.path.join(synthetic_root, item)))
        original_embeddings.append(original)
        synthetic_embeddings.append(synthetic)

    original_pairs = [
        cosine_similarity([original_embeddings[i]], [original_embeddings[j]])[0][0]
        for i, j in combinations(range(len(original_embeddings)), 2)
        if original_embeddings[i] is not None and original_embeddings[j] is not None
    ]
    synthetic_pairs = [
        cosine_similarity([synthetic_embeddings[i]], [synthetic_embeddings[j]])[0][0]
        for i, j in combinations(range(len(synthetic_embeddings)), 2)
        if synthetic_embeddings[i] is not None and synthetic_embeddings[j] is not None
    ]
    cross_pairs = [
        cosine_similarity([original], [synthetic])[0][0]
        for original in original_embeddings
        if original is not None
        for synthetic in synthetic_embeddings
        if synthetic is not None
    ]

    return {
        "original": np.asarray(original_pairs),
        "synthetic": np.asarray(synthetic_pairs),
        "cross": np.asarray(cross_pairs),
    }
