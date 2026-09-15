# FreePrompt baseline

FreePrompt is an external image-editing baseline used to generate paired
training images. This repository does not copy its sampler or checkpoints.

The paper settings are recorded in [`config.example.yaml`](config.example.yaml):
an empty source prompt, a 0.4 self-replace ratio, 512x512 images, 50 sampling
steps, and classifier-free guidance 7.5. Run the upstream FreePrompt code with
the source/private caption and public caption columns from the project
manifest.
