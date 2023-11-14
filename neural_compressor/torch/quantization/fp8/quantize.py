import torch
import copy
import os
from neural_compressor.torch.quantization.utils import set_module
from ..modules import BatchMatmul, Matmul


white_list = [torch.nn.Linear, BatchMatmul, Matmul]


def quantize_dynamic(model, inplace=True):
    q_model = model if inplace else copy.deepcopy(model)
    from neural_compressor.torch.quantization.fp8.modules import FP8DynamicLinear
    for n, m in q_model.named_modules():
        if isinstance(m, torch.nn.Linear):
            new_m = FP8DynamicLinear(m, use_amax=True)
            set_module(q_model, n, new_m)
    return q_model


def _add_observer(model, algorithm='minmax'):
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
    ### Insert input observer into model, only for fp8_e4m3 static quantization ###
    from .observer import MinMaxObserver, FP8HistogramObserver
    for name, module in model.named_modules():
        if isinstance(module, tuple(white_list)):
            module.add_module(
                'input_activation_post_process', FP8HistogramObserver() if \
                            algorithm == 'kl' else MinMaxObserver()
            )
        if isinstance(module, (BatchMatmul, Matmul)):
            module.add_module(
                'input_activation_post_process1', FP8HistogramObserver() if \
                        algorithm == 'kl' else MinMaxObserver()
            )
        module.register_forward_pre_hook(input_observer_forward_pre_hook)


def prepare(model, qconfig):
    _add_observer(model, algorithm=qconfig.act_algo)
    return model

def _remove_observer(model):
    for name, module in model.named_modules():
        HF_max = 240 if os.getenv('PT_USE_FP8_143') is not None else 57344
        if hasattr(module, 'input_activation_post_process'):
            if hasattr(module.input_activation_post_process, '_non_linear_param_search'):  # kl
                min_val, max_val = module.input_activation_post_process._non_linear_param_search()
            else:
                min_val = module.input_activation_post_process.min_val
                max_val = module.input_activation_post_process.max_val
            amax = torch.max(torch.abs(max_val), torch.abs(min_val))
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

def _replace_module(model):
    from neural_compressor.torch.quantization.fp8.modules import FP8Linear
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module = FP8Linear(module)
            set_module(model, name, module)


def convert(model):
    _remove_observer(model)
    _replace_module(model)
    return model


def quantize(model, qconfig, calib_func, inplace=True):
    q_model = model if inplace else copy.deepcopy(model)
    q_model = prepare(q_model, qconfig)
    calib_func(q_model)
    q_model = convert(q_model)
    return q_model


# def autotune(fp32_model, quant_config, tune_config, eval_func, ...):
#     pass