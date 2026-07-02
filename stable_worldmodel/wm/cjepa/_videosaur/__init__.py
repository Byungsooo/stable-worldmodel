"""Vendored subset of martius-lab/videosaur (MIT License, see ./LICENSE).

VideoSAUR: Zadaianchuk & Seitzer et al., "Object-Centric Learning for
Real-World Videos by Predicting Temporal Feature Similarities", NeurIPS 2023.
https://github.com/martius-lab/videosaur

Trimmed to the inference-time modules needed to reconstruct the frozen
PushT VideoSAUR encoder used by C-JEPA (arXiv:2602.11389) and load its
released checkpoint (huggingface.co/HazelNam/CJEPA). Training-only
machinery (YAML-driven module builders, losses, decoders, the Lightning
training loop) has been dropped; class names, submodule structure, and
parameter names are kept identical to the upstream source so that
checkpoints trained with the original code load via ``load_state_dict``
without key remapping.
"""
