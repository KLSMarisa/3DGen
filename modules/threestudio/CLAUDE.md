# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Quick commands

- Install deps (recommended in a venv):
  - python3 -m pip install --upgrade pip
  - python3 -m pip install -r requirements.txt
  - python3 -m pip install -r requirements-dev.txt (for lint/type checks)
- Run training / testing / export (examples):
  - python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a hamburger"
  - python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a baby bunny"
  - python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
- Launch Gradio UI locally:
  - python gradio_app.py launch
  - For public listening: python gradio_app.py launch --listen
- Run a single test or lint:
  - pytest path/to/test_file.py::test_name
  - mypy
  - pylint <package/module>
  - black --check .

High-level architecture

- Purpose: threestudio is a unified framework to lift 2D text-to-image models into 3D content creation (text/image-to-3D). It implements multiple pipelines (DreamFusion, Magic3D, ProlificDreamer, SDI, Zero-1-to-3, etc.) and export tools (mesh/obj, video).

- Entry points:
  - launch.py: main CLI for training/validation/testing/export. It parses an OmegaConf YAML config, constructs the data module and the system, and uses PyTorch Lightning Trainer to run.
  - gradio_app.py: launches the web UI for interactive use.

- Configuration system:
  - Uses OmegaConf dataclasses. Configs live under configs/*.yaml. launch.py loads a config via utils.config.load_config and merges CLI overrides (you can pass parsed_yaml keys as additional CLI args without `--`).
  - Typical runtime config: ExperimentConfig contains `data` (data module config), `system` (system config), `trainer` (PL trainer args), `checkpoint` and `trial_dir`.

- Core abstractions
  - BaseSystem (threestudio/systems/base.py): all methods (DreamFusion, Magic3D, etc.) subclass BaseSystem. A System composes modules: geometry, material, background, renderer, guidance, prompt_processor, exporter.
  - BaseModule / BaseObject (threestudio/utils/base.py): modules and objects used by systems. Modules can be Updateable to perform per-step updates.
  - DataModule: found in threestudio/data or similar; they provide training/validation dataloaders and camera sampling.
  - Guidance & PromptProcessor: provide gradients (guidance) and text embeddings (prompt_processor) for SDS-style optimization.

- Geometry / Material / Renderer separation
  - Geometry: implicit (NeRF/SDF/volume) or explicit (voxel grid, tetrahedra). Configs control encoding (tiny-cuda-nn), MLP heads, isosurface extraction.
  - Material: maps geometry features to colors (PBR, neural-radiance-material, latent adapters).
  - Renderer: volume renderers (nerf-volume, neus-volume), rasterizers (nvdiffrast) and patch-renderer for memory optimization.
  - Exporters: mesh-exporter (obj/obj-mtl), with options for UV unwrapping via xatlas and texture baking.

- Guidance models and supported backends
  - Stable Diffusion (diffusers), DeepFloyd IF, Zero123 weights, and experimental LoRA/SDXL usage. Guidance configs (system.guidance.*) control model paths and memory optimizations (attention slicing, channel formats, cpu-offload).

- Training lifecycle
  - launch.py builds datamodule and system, constructs PL Trainer and callbacks (checkpointing, logging, config snapshot). When training, checkpoints are saved to [trial_dir]/ckpts.
  - Resume/weights: use `resume=path/to/ckpt` to continue training or `system.weights=` to only load weights without optimizer state.

Important repository notes for automation

- Large assets and models are excluded in .gitignore (*.ckpt, outputs/). Avoid attempting to download or commit large model files automatically. When automating, expect that pre-trained weights must be placed under load/ (e.g., load/zero123 or load/zero123/stable-zero123.ckpt).
- Many configs assume an NVIDIA GPU and CUDA installed. CI or bots should gracefully skip GPU-only runs or run small smoke configs.
- Custom modules: the `custom/` directory is scanned and imported by launch.py. Be cautious when running custom code; import failures are printed but do not stop the run.

Files and places to check when editing

- launch.py: CLI and trainer orchestration (threestudio:line 104-174, 228-236)
- utils/config.py: configuration parsing and ExperimentConfig dataclass
- threestudio/systems/: system implementations; change here when adding algorithms
- configs/: YAML experiment templates and defaults
- gradio_app.py: web UI entrypoint
- load/: expected location for external weights, images and prompt library

When to enter Plan Mode / ask for confirmation

- Use EnterPlanMode before making multi-file changes that affect model interfaces, training loops, or config dataclasses.
- Ask user confirmation before: committing large binary files, changing default checkpoint behavior, running long GPU jobs, or modifying CI.

Additions to memory or hooks

- No CLAUDE-specific hooks were found. If you want persistent automation (pre-run checks, model download hooks), add repository hooks in settings.json using the update-config skill.

That's it.