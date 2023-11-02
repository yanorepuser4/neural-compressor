from neural_compressor.common.config import (
    _StaticQuantConfig,
    _SmoothQuantConfig
    )

from typing import Union, Any, List

class StaticQuantConfig(_StaticQuantConfig):
    tunable_params = ['act_dtype', 'act_sym', 'weight_dtype', 'weight_sym']
    def __init__(
        self,
        act_dtype: Union[Any, List[Any]] = None,
        act_sym: Union[bool, List[bool]] = None,
        weight_dtype: Union[Any, List[Any]] = None,
        weight_sym: Union[bool, List[bool]] = None,
        white_list: List[Any] = None,
        black_list: List[Any] = None,
        ) -> None:
        super().__init__(
            act_dtype = act_dtype,
            act_sym = act_sym,
            weight_dtype = weight_dtype,
            weight_sym = weight_sym,
            white_list = white_list,
            black_list=black_list,
        )


class SmoothQuantConfig(_SmoothQuantConfig):
    tunable_params = ['alpha', 'folding']
    def __init__(
        self,
        alpha: Union[float, List[float]] = None,
        folding: Union[bool, List[bool]] = None,
        ) -> None:
        super().__init__(alpha=alpha, folding=folding)


import torch
static_config_common = StaticQuantConfig(
    act_dtype=[torch.int8, torch.uint8],
    act_sym=[True, False],
    weight_dtype=[torch.int8, torch.uint8],
    weight_sym=[True, False],
    white_list=[torch.nn.Conv2d, torch.nn.Linear],
    )

static_config_add = StaticQuantConfig(
    act_dtype=[torch.int8, torch.uint8],
    act_sym=[True, False],
    white_list=[torch.add],
    )

sq_config = SmoothQuantConfig(alpha=[0.1], folding=[True, False])


print(SmoothQuantConfig.tunable_params)


# # ###############
# # ## End user
# # ###############
# import torch
# from neural_compressor.torch.quantization.config import StaticQuantConfig
# static_config = StaticQuantConfig(act_dtype=[torch.int8, torch.float32], act_sym=True)
# from neural_compressor.torch.quantization import autotune
# user_fp32_model = UserModel()
# q_model = autotune(user_fp32_model, static_config, eval_func=...)