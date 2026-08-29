Apocynthion is building AlphaMotion, a cerebellum-like motion foundation model for robots. Inspired by AlphaFold's pair representation with triangle updates, it learns joint coordination and temporal dynamics, and uniquely recovers full-body motion from partial or egocentric video—unlocking internet-scale human video as training data.

Founded by a former RAI Institute (Marc Raibert's lab) researcher and a Yahoo AI engineer, with team from NVIDIA GEAR, DeepMind, Boston Dynamics, and Stanford/Berkeley/CMU PhDs.

Products: ArenaLabs, an API and preview platform with native Isaac Lab integration (6 months), then ArenaOS, a motion-control kernel for Jetson-based OEMs—the cerebellum beneath the VLA/GR00T cortex.

AlphaMotion-1 was trained on RTX PRO 6000 Blackwell workstations, one signed by Jensen at GTC. AlphaMotion-2 will scale to 128×H100 via DGX Cloud / CoreWeave / Lambda, using Cosmos for data, Isaac Lab/Sim for simulation, and Jetson + TensorRT-LLM + Triton for deployment.
