# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

People who want to make a photo safer to share and people who want to create playful or more representative remixes. These are equally important audiences.

## Product Purpose

Let a person choose or take a photo, choose a quick edit mode, and receive an edited image they can preview and download.

## Positioning

One simple photo-editing flow serves both privacy edits and creative remixes through quick presets.

## Capabilities and Constraints

- The local demo accepts a photo upload or mobile camera capture and returns a Gemini-generated image.
- The Gemini API key is held by the server. The current demo does not save uploaded images to disk.
- The three quick edit modes are: a Halloween-inspired privacy edit, a more diverse depiction of people, and a location remix that moves the same moment to an unexpected setting. The funny remix idea was dropped because it overlaps the broader creative remix.

## Evidence on Hand

- The existing demo implementation is in `demo/app.py`.
- No representative production images, user research, or customer evidence have been supplied.

## Product Principles

- Make privacy and creative editing equally easy to reach.
- Let people choose a clear edit mode before sending the photo.
- Keep the original and generated result easy to distinguish, preview, and download.
