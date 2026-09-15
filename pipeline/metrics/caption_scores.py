"""Compute captioning utility scores used in the paper."""


def compute_caption_scores(predictions, references):
    """Return BLEU-4 and CIDEr for caption predictions and references."""
    import evaluate

    bleu = evaluate.load("bleu").compute(predictions=predictions, references=references, max_order=4)
    cider = evaluate.load("cider").compute(predictions=predictions, references=references)
    return {"bleu4": bleu["bleu"], "cider": cider["cider"]}
