import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_directional_clip_score(model, processor, original_images, edited_images, original_captions, edit_captions):
    """Compute directional CLIP similarity for original and edited image batches."""
    device = next(model.parameters()).device

    images = torch.cat([original_images, edited_images], dim=0).to(device)
    image_features = model.get_image_features(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    batch_size = len(original_images)
    original_features = image_features[:batch_size]
    edited_features = image_features[batch_size:]

    original_inputs = processor(text=list(original_captions), return_tensors="pt", padding=True, truncation=True).to(device)
    edit_inputs = processor(text=list(edit_captions), return_tensors="pt", padding=True, truncation=True).to(device)
    original_text_features = model.get_text_features(**original_inputs)
    edit_text_features = model.get_text_features(**edit_inputs)
    original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)
    edit_text_features = edit_text_features / edit_text_features.norm(dim=-1, keepdim=True)

    image_direction = edited_features - original_features
    text_direction = edit_text_features - original_text_features
    return F.cosine_similarity(image_direction, text_direction)
