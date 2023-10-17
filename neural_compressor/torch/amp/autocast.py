from typing import Any, Optional

import os
import torch
from torch.types import _dtype
from neural_compressor.torch.dtype import float8_e4m3, float8_e5m2


class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

        # Enables autocasting for the inference pass
        with torch.autocast(device_type="hpu", dtype=torch.float8_e4m3):
            output = model(input)

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @torch.autocast(device_type="cuda")
            def forward(self, input):
                ...         

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(str, required):  Device type to use. Possible values are: 'cuda', 'cpu', 'xpu' and 'hpu'.
                                     The type is the same as the `type` attribute of a :class:`torch.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(torch_dtype, optional):  Whether to use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        self.device = device_type
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled
        if device_type == "hpu" and dtype in [float8_e4m3, float8_e5m2]:
            if dtype == float8_e4m3:
                os.environ["PT_USE_FP8_143"] = str(1)
            else:
                os.environ.pop("PT_USE_FP8_143", None)
        else:
            self._autocast = torch.autocast(device_type, dtype, enabled, cache_enabled)

    def __enter__(self) -> None:
        if self.device == "hpu" and self.fast_dtype in [float8_e4m3, float8_e5m2]:
            from neural_compressor.torch.amp.modules.fp8_functions import replace_func
            # This function will replace F.linear and torch.matmul with the fp8 one
            replace_func()
        else:
            self._autocast.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.device == "hpu" and self.fast_dtype in [float8_e4m3, float8_e5m2]:
            from neural_compressor.torch.amp.modules.fp8_functions import recover_func
            # This function will recover F.linear and torch.matmul with the original one
            recover_func()
        else:
            self._autocast.__exit__(exc_type, exc_value, traceback)