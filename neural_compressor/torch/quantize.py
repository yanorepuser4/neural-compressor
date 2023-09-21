from collections import OrderedDict
import torch
from mpemu.module_wrappers import BatchMatmul, Matmul, AddMatmul, EltwiseMul, EltwiseAdd, EltwiseDiv
from .observer import FP8PercentileObserver, FP8HistogramObserver, MinMaxObserver

fp8_white_list = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag,
                  BatchMatmul, Matmul, AddMatmul, EltwiseMul, EltwiseAdd, EltwiseDiv]

def _prepare_observer(model, qconfig_dict):
    def input_observer_forward_pre_hook(self, input):
        try:
            if isinstance(input[0], torch.Tensor):
                self.input_activation_post_process(input[0])
            if hasattr(self, 'input_activation_post_process1') and isinstance(input[1], torch.Tensor):
                self.input_activation_post_process1(input[1])
            return input
        except Exception as e:
            # The KL algorithm may encounter a overflow error on EltwiseAdd.
            pass

    for name, module in model.named_modules():
        if name in qconfig_dict and qconfig_dict[name]['activation']['dtype'] in ['fp8_e4m3', 'fp8_e3m4']:
            algorithm = qconfig_dict[name]['activation']['algorithm']
            qtconfig = qconfig_dict[name].iact_qconfig
            from mpemu.module_wrappers import (BatchMatmul, Matmul, AddMatmul,
                                                EltwiseAdd, EltwiseMul, EltwiseDiv)
            module.add_module(
                'input_activation_post_process', FP8HistogramObserver(qtconfig=qtconfig) if \
                            algorithm == 'kl' else MinMaxObserver()
            )
            if type(module) in [BatchMatmul, Matmul, AddMatmul,
                                EltwiseAdd, EltwiseMul, EltwiseDiv]:
                module.add_module(
                    'input_activation_post_process1', FP8HistogramObserver(qtconfig=qtconfig) if \
                            algorithm == 'kl' else MinMaxObserver()
                )
            module.register_forward_pre_hook(input_observer_forward_pre_hook)

def _calibration_for_scale(model, model_qconfig_dict):
    r"""post process after calibration for scale calcuation

    """
    def _get_combine_scale(amax, mean_val, HF_max, HF_min):
        scale = HF_max / amax
        if self.scale_method == 'mean':
            mean_val = mean_val if mean_val > 1e-6 else HF_min
            if 0.0 < mean_val < HF_min:
                scale_mean = HF_min / mean_val
            else:
                scale_mean = torch.tensor(1.0)
            # make sure amax is included in new scale range.
            if scale_mean < scale:
                scale = scale_mean
        return scale

    for name, module in model.named_modules():
        if hasattr(module, 'input_activation_post_process'):
            HF_max = model_qconfig_dict[name].iact_qconfig.get_flt_max()
            HF_min = model_qconfig_dict[name].iact_qconfig.get_flt_min()
        if hasattr(module, 'input_activation_post_process'):
            if hasattr(module.input_activation_post_process, '_non_linear_param_search'):
                min_val, max_val = module.input_activation_post_process._non_linear_param_search()
            else:
                min_val = module.input_activation_post_process.min_val
                max_val = module.input_activation_post_process.max_val
            amax = torch.max(torch.abs(max_val), torch.abs(min_val))
            if hasattr(module.input_activation_post_process, "mean_val") \
              and self.scale_method == 'mean':
                mean_val = module.input_activation_post_process.mean_val
                scale = _get_combine_scale(amax, mean_val, HF_max, HF_min)
            else:
                scale = HF_max / amax
            module.register_parameter('scale', torch.nn.Parameter(scale))
            delattr(module, 'input_activation_post_process')
        if hasattr(module, 'input_activation_post_process1'):
            if hasattr(module.input_activation_post_process1, '_non_linear_param_search'):
                min_val, max_val = module.input_activation_post_process1._non_linear_param_search()
            else:
                min_val = module.input_activation_post_process1.min_val
                max_val = module.input_activation_post_process1.max_val
            amax = torch.max(torch.abs(max_val), torch.abs(min_val))
            if hasattr(module.input_activation_post_process1, "mean_val") \
              and self.scale_method == 'mean':
                mean_val = module.input_activation_post_process1.mean_val
                scale = _get_combine_scale(amax, mean_val, HF_max, HF_min)
            else:
                scale = HF_max / amax
            module.register_parameter('scale1', torch.nn.Parameter(scale))
            delattr(module, 'input_activation_post_process1')

        # remove observer hooks
        hook_map = module._forward_pre_hooks
        handle_ids_to_remove = set()
        for handle_id, hook_fn in hook_map.items():
            if hasattr(hook_fn, '__name__') and \
              hook_fn.__name__ == 'input_observer_forward_pre_hook':
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

def add_quantization_hooks(model, model_qconfig_dict):
    def quantize_module_inputs(self, input):
        if self.qconfig.iact_qconfig is not None and self.qconfig.iact_qconfig.is_enabled:
            input_q = []
            scale = None
            if hasattr(self, 'scale'):
                scale = self.scale
            for i in range(len(input)):
                # for Matmul and BatchMatmul
                if i == 1 and hasattr(self, 'scale1'):
                    scale = self.scale1
                tensor = input[i]
                if scale and torch.isinf(scale):
                    scale = torch.tensor(3.4E38)
                tensor_q = _quantize_tensor(tensor, self.qconfig.iact_qconfig, scale=scale)
                input_q.append(tensor_q)

            input = tuple(input_q)
        return input

    hook_handles = []
    for name, module in model.named_modules():
        if name in model_qconfig_dict:
            handle_pf = module.register_forward_pre_hook(quantize_module_inputs)
            hook_handles.append(handle_pf)
    return hook_handles

# FP8: Single entry function for quantizing tensor.
def _quantize_tensor(tensor, qtconfig, scale=None, inplace=False):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    qtconfig.check_validity(qtconfig.dtype, qtconfig.scheme)
    mode = qtconfig.dtype.upper()

    if not scale:
        scale = get_fp8_scale(tensor, qtconfig)

    if qtconfig.scheme is not None:
        mode += "_"+qtconfig.scheme.upper()

    if tensor.device.type == 'cuda':
        from mpemu.pytquant.cuda import fpemu_cuda as fpemu_cpp
    else:
        from mpemu.pytquant.cpp import fpemu_cpp
    try:
        tensor_q = fpemu_cpp.FPEmuOp.apply(tensor, mode, inplace, scale)
    except:
        # for dlrm, tensor is too large
        split_parts = 10
        tensor_q = None
        num = int(tensor.shape[0] / split_parts)
        for i in range(split_parts):
            if i ==0:
                tensor_q = fpemu_cpp.FPEmuOp.apply(tensor[:num], mode, True, scale)
            elif i == split_parts - 1:
                tensor_q = torch.cat((tensor_q, fpemu_cpp.FPEmuOp.apply(tensor[i*num:], mode, True, scale)), 0)
            else:
                tensor_q = torch.cat((tensor_q, fpemu_cpp.FPEmuOp.apply(tensor[i*num:(i+1)*num], mode, True, scale)), 0)
    return tensor_q


def quantize_model_weights(model, model_qconfig_dict, force_granularity=None):
    def _quantize_weight(module, wt_qconfig, granularity='per_channel'):
        if granularity == 'per_channel':
            # this would be innerloop parallelized.
            for i, data in enumerate(module.weight.data):
                module.weight.data[i] = _quantize_tensor(module.weight.data[i], wt_qconfig)
        else:
            module.weight.data = _quantize_tensor(module.weight.data, wt_qconfig)
        module.weight.data.copy_(module.weight.data)

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and name in model_qconfig_dict:
            qconfig = model_qconfig_dict[name]
            if force_granularity:
                granularity = force_granularity
            else:
                granularity = qconfig.w_quantizer.granularity
            _quantize_weight(module, qconfig.wt_qconfig, granularity)

def _create_model_qconfig(model, qconfig):
    r"""Internal function to create a model_qconfig per op.

    Args:
        qconfig: a instance of QuantConfig

    """
    assert(isinstance(qconfig, QuantConfig))
    module_dict = dict(model.named_modules())
    for op_name, child in module_dict.items():
        if type(child) in self.white_list and op_name not in exempt_modules:
            quantizable_ops.append(
                (op_name, unify_op_type_mapping[str(child.__class__.__name__)]
                 if str(child.__class__.__name__) in unify_op_type_mapping else str(
                     child.__class__.__name__)))



def quantize(model, qconfig, run_fn, run_args, mapping=None, inplace=False):
    r"""Quantize the input float model with post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        model: input float model
        qconfig: a dict or instance of QuantConfig
        run_fn: a calibration function for calibrating the prepared model
        run_args: positional arguments for `run_fn`
        mapping: correspondence between original module types and quantized counterparts
        inplace: carry out model transformations in-place, the original module is mutated

    Return:
        Quantized model.
    """
    model_qconfig_dict = _create_model_qconfig(qconfig)

    if run_fn:
        _prepare_observer(model, model_qconfig_dict)
        run_fn(model, *run_args)
        _calibration_for_scale(model, model_qconfig_dict)

    # add fp8 emulation hook
    from mpemu import qutils
    qutils.reset_quantization_setup(model, model_qconfig_dict)
    quantize_model_weights(model, model_qconfig_dict) # inplace
    add_quantization_hooks(model, model_qconfig_dict)

    model.qconfig = qconfig
    return model

def quantize_dynamic(model, qconfig, mapping=None, inplace=False):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig: a dict or instance of QuantConfig
        mapping: correspondence between original module types and quantized counterparts
        inplace: carry out model transformations in-place, the original module is mutated

    """
    return quantize(model, qconfig, run_fn=None, run_args=None, mapping=mapping, inplace=inplace)
