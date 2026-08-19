# Optional Upstream Repositories

UniFur does not vendor external research code. Clone only the baseline or prior
required by the experiment into this ignored directory:

- HairGS
- GaussianHaircut
- NeuralFur
- Im2Haircut
- SAM 3D Objects (optional single-view initialization prior)

Keep each upstream repository's original license and install it in its own
environment. See `EXTERNAL_BASELINES.md` and the corresponding setup script for
the expected local path and protocol.
