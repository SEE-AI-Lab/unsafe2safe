# External dependencies

The public code keeps large model repositories, checkpoints, datasets, and
generated outputs outside this repository. Install the external projects
needed for the workflow you use and follow their licenses and model terms.

- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) provides
  the external diffusion trainer used by `pipeline/stage2/`.
- [OminiControl](https://github.com/Yuanshi9815/OminiControl) provides the
  external FLUX adapter used by `pipeline/adapters/ominicontrol/`.
- [FlowEdit](https://github.com/fallenshock/FlowEdit) provides the sampler used
  by `pipeline/adapters/flowedit/`.
- [LAVIS](https://github.com/salesforce/LAVIS) provides the external BLIP-2
  captioning trainer used by `pipeline/adapters/lavis/`.
- [Face Anon Simple](https://github.com/hanweikung/face_anon_simple) provides
  the optional face-anonymization implementation used by its adapter.
- FreePrompt, DeepPrivacy2, and ImageMAE remain external baselines or trainers;
  their handoff settings are recorded in `pipeline/adapters/`.

The adapter READMEs contain setup details where an external checkout is
required. The example paths and settings are not a guarantee that every
upstream version is compatible with every environment.
