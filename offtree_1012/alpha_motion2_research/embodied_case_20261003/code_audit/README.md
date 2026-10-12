# AlphaMotion1 actual-forward evidence assets

These five figures use saved tensors from strict-loaded L_M20k_0903: G1 fallAndGetUp3_subject1, frame 1800 (30 fps), to a fresh descriptor built from the official Unitree H2 MJCF. No model was run or modified by this plotting script.

- 01-source-token256: all 256 channels of the actual left-elbow pose projection, body descriptor and combined encoder input; common signed scale.
- 02-encoder-attention: layer 0 / head 0, 36 G1 queries by 36 G1 keys.
- 03-rotation-pool-attention: head 0, 128 learned rotation queries by 36 G1 keys.
- 04-target-decoder-attention: layer 0 / head 0, 36 official H2 queries by 128 rotation latents.
- 05-fsq13: actual rotation slot 0 before and after one-stage scalar quantization. Raw and normalized values use explicitly different axes.

Attention heatmaps retain every cell and share a linear probability scale from 0 to 1. Orange outlines select one query; lower bars show that row on a separately labeled probability axis. Sparse axes use zero-based indices; the manifest preserves all joint names. Stored attention was reconstructed from forward-captured inputs and checkpoint projections, with module-output parity below 7.2e-7. These weights do not establish causal importance or joint correspondence.

hpos-semantics.json supplements the original trace with source hashes, exact line snippets and checkpoint/config evidence. Target-root channel 0 has a direct ground-relative-height loss in meters and a temporal height-difference loss. Root channels 1-2 and nonroot raw hpos have no direct metric XYZ supervision under this configuration, but enter re-encoding consistency. Separate FK is target-root-relative centimeters from predicted rotations and actual target offsets. Raw hpos is not substituted for FK.

plot-manifest.json hashes every PNG/SVG and links source tensor keys. selected-tensor-values.json preserves plotted strips and attention rows. SVG text stays editable. render_evidence.py reproduces all outputs from the adjacent alpha1 trace.json and trace.npz with NumPy and Matplotlib.

Scope: three independent frames exist in the parent trace; these assets use its first one. This is a training-domain single-frame kinematic demonstration, not temporal generation, physical-robot control, joint-limit or physics validation, or held-out generalization. Official H2 descriptors do not bind the legacy anonymous cache tree to official meshes. No Google Doc or source/model/service state was changed.
