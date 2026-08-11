# Checkpoint Layout

Checkpoint contents are local runtime assets and are ignored by Git.

Expected directories:

```text
checkpoints/
|-- sam3d/
|   `-- hf/
|       |-- pipeline.yaml
|       |-- ss_generator.ckpt
|       |-- slat_generator.ckpt
|       |-- ss_decoder.ckpt
|       |-- slat_decoder_gs.ckpt
|       |-- slat_decoder_gs_4.ckpt
|       `-- slat_decoder_mesh.ckpt
`-- mocap_anything/
    |-- RMBG-1.4/
    |-- TripoSG/
    |-- video2pose/
    `-- video2pose2rot/
```

The current WSL integration directory contains copied local checkpoints in
this layout.
