# Third-party code and provenance

Unsafe2Safe keeps large upstream repositories outside this checkout. The
project code imports them at runtime and does not publish copied upstream
source, checkpoints, datasets, or generated outputs.

## InstructPix2Pix

- Repository: [timothybrooks/instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix)
- Reproduction revision: `0dffd1e`
- Install: clone the repository, check out that revision, and pass its path
  through `INSTRUCT_PIX2PIX_ROOT` or the training launcher.
- License: see the upstream [LICENSE](https://github.com/timothybrooks/instruct-pix2pix/blob/main/LICENSE).
  The upstream repository also contains components derived from Stable
  Diffusion, so follow the notices and terms in that checkout as well.

`pipeline/instruct_pix2pix.py`, `pipeline/safe_attention.py`, and
`pipeline/unsafe2safe_model.py` are the Unsafe2Safe-specific import and model
adapters. They leave the external checkout unchanged.

## OminiControl

- Repository: [Yuanshi9815/OminiControl](https://github.com/Yuanshi9815/OminiControl)
- Reproduction revision: `65d929e`
- Paper-era revision also supported by the adapter: `54913bc`
- Install: clone the repository, check out one selected revision, install its
  requirements, and set `OMINICONTROL_ROOT` as described in
  [`pipeline/ominicontrol/README.md`](pipeline/ominicontrol/README.md).
- License: upstream [Apache-2.0 LICENSE](https://github.com/Yuanshi9815/OminiControl/blob/main/LICENSE).

`pipeline/ominicontrol/` contains only the Unsafe2Safe dataset mapping and
launch/generation wrappers.

## FlowEdit

- Repository: [fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit)
- Reproduction revision: `2620a9364f0f9c21368c36266e6200ff250273ec`
- Install: clone the repository, check out that revision, install the
  upstream-compatible diffusion dependencies, and set `FLOWEDIT_ROOT` as
  described in [`pipeline/flowedit/README.md`](pipeline/flowedit/README.md).
- License: upstream [MIT LICENSE](https://github.com/fallenshock/FlowEdit/blob/main/LICENSE).

`pipeline/flowedit/` contains only the Unsafe2Safe CSV condition mapping and
portable generation wrapper. The upstream sampler, assets, data, and model
weights are not copied into this repository.

## BLIP-2 and LAVIS

- Repository: [salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- Reproduction revision: `baad2d7c8df599d8d9b081ba2e946626eaa2dc34` (tag
  `v1.0.2`)
- Install: clone that revision separately and install it in the dedicated
  BLIP-2 environment described in [`pipeline/README.md`](pipeline/README.md).
- License: LAVIS source is distributed under the upstream BSD-3-Clause
  license.  BLIP-2, OPT, FLAN-T5, and their checkpoints retain their own
  upstream terms and model-card restrictions.

The historical LAVIS copy in `temppp_donttouch/LAVIS/` is not part of this
release.  Its project-specific COCO image routing was extracted into
`pipeline/blip2_captioning.py`; its hard-coded dataset paths, GPU/cache
settings, and patched LAVIS internals were replaced by explicit arguments and
the pinned external dependency.

## ImageMAE dataset boundary

The project-specific downstream dataset definition is kept in
[`pipeline/image_mae_dataset.py`](pipeline/image_mae_dataset.py). It is
separate from the diffusion dataset and contains no ImageMAE model code. The
ImageMAE model/trainer remains external; the former local clone, notebooks,
and experimental classifier code are not part of the public release.

## Unsafe2Safe license status

This repository currently has no top-level project `LICENSE` file. Do not
assume a project license for the Unsafe2Safe-specific code until the authors
choose and add one. Third-party dependencies retain their own licenses.
