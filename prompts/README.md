# Published prompts

The canonical Stage 1 prompts follow Unsafe2Safe, Appendix S6.1:
https://arxiv.org/html/2603.28605v1#A6.SS1

| File | Paper figure |
| --- | --- |
| `privacy_criteria.txt` | S7: privacy criteria |
| `intern_image_flagging.txt` | S8: privacy filtering |
| `intern_image_captioning.txt` | S9: privacy review, private caption, public caption |
| `qwen_image_captioning-private-expand.txt` | S10: edit instruction generation |
| `qwen_caption_combine.txt` | S11: public caption and edit instruction combination |
| `intern_image_flagging-custom.txt` | S12: custom privacy filtering |
| `intern_text_extraction.txt` | S13: visible text extraction |
| `intern_demographic_analysis.txt` | S14: demographic analysis |
| `intern_image_flagging-compare.txt` | S15: anonymization evaluation user prompt |
| `intern_image_flagging-compare-system.txt` | S15: anonymization evaluation system prompt |

These prompts reproduce the published wording with
line breaks normalized and `[CRITERIA]` expanded using the categories in Figure S7.
The edit prompt retains the paper's Stable Diffusion wording and spelling.
`{public_caption}`, `{edit_caption}`, and `{image_class}` are runtime template
variables where applicable. The caption prompt does not require a class input.
The custom filtering prompt uses `{criteria}` and `{criteria_description}` for
the paper's criterion placeholders. The evaluator loads the full Figure S15
system prompt by default; `--system-prompt` remains available as an override.

Figure S9 permits anonymous presence such as “a person”; it does not require
removing a whole subject merely because identifying attributes were flagged.
Both captions retain the published 50-token limit. Figure S10 requests at least
three appearance changes per person and limits the instruction to 45 words.
The edit-instruction stage is conditioned on the public caption, without passing
the original image or privacy review to the LLM.

The existing Stage 1 parser supports the published `PRIVATE_CAPTION` and
`PUBLIC_CAPTION` section names as well as the older section-name aliases.
Applications that need JSON responses can map these sections to a response
schema without changing the substantive captioning or editing instructions.

Source: Minh Dinh and SouYoung Jin, *Unsafe2Safe: Controllable Image
Anonymization for Downstream Utility*, arXiv:2603.28605v1 (CC BY 4.0).
