import torch

class autocast(torch.autocast):
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
        if device_type == "hpu":
            assert dtype == torch.float8_e4m3 or dtype == torch.float8_e5m2, "autocast only supports float8 e4m3 and e5m2 formats."
            ## TODO: add INC's fp8 implementation here ##
            pass
        else:
            torch.autocast(device_type, dtype, enabled, cache_enabled)
