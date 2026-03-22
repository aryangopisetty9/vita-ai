"""
Vita AI – Backend Package
==========================
This __init__.py sets critical environment variables BEFORE any other
imports.  In particular ``KERAS_BACKEND=jax`` must be set before Keras
or TensorFlow is imported so that the open-rppg package (which uses
Keras 3 + JAX) does not collide with TensorFlow's own Keras backend.
"""

import os as _os

# ── JAX / Keras backend isolation ────────────────────────────────────────
# Open-rPPG uses Keras 3 with the JAX backend.  TF 2.21 ships with Keras 3
# as well.  If TF is imported first, Keras initialises with the TF backend
# and subsequent JAX-backed inference fails with:
#   "EagerTensor is not a valid JAX type"
# Setting the env-var *here* (the very first import in the package) ensures
# it is effective before either Keras or TF is loaded.
_os.environ.setdefault("KERAS_BACKEND", "jax")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
