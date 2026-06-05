"""Skip PyTorch's default weight init while building a model that a checkpoint
is about to overwrite.

``Ideogram4Pipeline.from_pretrained`` constructs the full ~4.6B-param
transformer out of ordinary ``nn.Linear`` layers, so every layer runs
``kaiming_uniform_`` on the CPU during ``__init__``. That init is
single-threaded; on a busy box it can spin one core for *minutes* before the
pretrained weights load on top and discard every one of those values.

``no_init_weights()`` no-ops the expensive per-element RNG initialisers for the
duration of the load, so construction costs ~nothing and the time goes to the
checkpoint read itself. Cheap deterministic inits (``zeros_``/``ones_``/
``constant_``) are left alone.

Only wrap a load that populates the whole model -- skipping init leaves
uninitialised memory in any parameter the checkpoint does *not* set.
"""

from __future__ import annotations

import contextlib

import torch.nn.init as _init

# Initialisers whose cost is the per-element RNG draw (the ones that show up
# pegging a core under py-spy). nn.Linear.reset_parameters uses kaiming_uniform_
# for weights and uniform_ for bias; embeddings use normal_.
_PATCHED = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform_",
    "xavier_normal_",
)


def _noop(tensor, *args, **kwargs):
    return tensor


@contextlib.contextmanager
def no_init_weights():
    """Temporarily replace the RNG-based ``nn.init`` functions with no-ops."""
    saved = {name: getattr(_init, name) for name in _PATCHED}
    for name in _PATCHED:
        setattr(_init, name, _noop)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(_init, name, fn)
