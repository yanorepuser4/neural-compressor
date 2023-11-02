from __future__ import annotations

"""
Algo/Kernel Capability:
1. Conv2d, Linear may have different capability.
2. cpu and gpu may have different capability.
3. ipex and stock pytorch may have different capability.

Expand: 
Per-Op-wise  (static quant)
Per-OpType-wise 
Per-Model-wise (sq)

Merge:
override Op-Type cap
override op cap

common:
- for end-user
- for tuning config
- for adaptor config

goal:
- new data type
- new algo
- customize the tuning order

merge:
    for sq_alpha, 
    for dtype,
    
expand:
    - no model info
    - have model info
"""


from typing import Dict, Union, Any, List



# 
# {
#     backend_name:
#         static:
# }


registered_configs = {}  # A dictionary to store registered configurations

def register_config(algo_name, priority):
    def decorator(cls):
        # Store the class in the registered_configs dictionary
        registered_configs[algo_name] = {
            'class': cls,
            'priority': priority
        }
        return cls  # Return the class unmodified
    return decorator


class AlgorithmConfig:

    def __init__(
        self,
        white_list: List[Any] = None,
        black_list: List[Any] = None,
        ) -> None:
        self.white_list = white_list
        self.black_list = black_list
        self.constrain_funcs = []

    def add_constraint_fun(self, new_filter_func):
        self.constrain_funcs.append(new_filter_func)

    @classmethod
    def expand(
        cls,
        merged: AlgorithmConfig,
        model_info: Dict = None
        ) -> AlgorithmConfig:
        """
        trial1:
            model_quant_config = {
                "op_type":{
                    'conv': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...),
                    'linear': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...)
                },
                "op_instance":{
                    'conv1': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...),
                    'linear3': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...)
                }
            }
        trial2:
            model_quant_config = {
                "op_type":{
                    'conv': AlgorithmConfig(act_dtype=int8, weight_dtype=int8, ...),
                    'linear': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...)
                },
                "op_instance":{
                    'conv1': AlgorithmConfig(act_dtype=int8, weight_dtype=int8, ...),
                    'linear3': AlgorithmConfig(act_dtype=int8, weight_dtype=uint8, ...)
                }
            }
            ...
        """
        pass

    @classmethod
    def merge(
        cls,
        user_config: AlgorithmConfig,
        fwk_config: AlgorithmConfig,
        model_info: Dict = None
        ):
        """
        For op_type
            {
                'conv': AlgorithmConfig(),
                'linear': AlgorithmConfig(),
            }
        For op_instance:
            {
                'conv':
                    'conv1': AlgorithmConfig()
                    'conv2': AlgorithmConfig(),
                'linear':
                    'linear1': AlgorithmConfig()
                    'conv2': AlgorithmConfig(),
            }
        """
        pass
    
    @classmethod
    def get_default_config(
        cls,
        model_info: Dict = None
        ) -> Dict[str: AlgorithmConfig]:
        """
        if model_info is not None:
            conv1: 
            conv2:
        else:
            conv:
            linear:
        """

@register_config(algo_name="static", priority=1)
class _StaticQuantConfig(AlgorithmConfig):
    tunable_params = ['act_dtype', 'act_sym', 'weight_dtype', 'weight_sym']
    def __init__(
        self,
        act_dtype: Union[Any, List[Any]] = None,
        act_sym: Union[bool, List[bool]] = None,
        act_granularity: Union[Any, List[Any]] = None,
        act_algorithm: Union[bool, List[bool]] = None,
        weight_dtype: Union[Any, List[Any]] = None,
        weight_sym: Union[bool, List[bool]] = None,
        weight_granularity: Union[Any, List[Any]] = None,
        weight_algorithm: Union[bool, List[bool]] = None,
        white_list: List[Any] = None,
        black_list: List[Any] = None,
        ) -> None:
        super().__init__(white_list=white_list, black_list=black_list)
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algorithm = act_algorithm
        self.weight_dtype = weight_dtype
        self.weight_sym = weight_sym
        self.weight_granularity = weight_granularity
        self.weight_algorithm = weight_algorithm


@register_config(algo_name="smooth_quant", priority=1)
class _SmoothQuantConfig(AlgorithmConfig):
    tunable_params = ['alpha', 'folding']
    def __init__(
        self,
        alpha: Union[float, List[float]] = None,
        folding: Union[bool, List[bool]] = None,
        ) -> None:
        super().__init__()
        self.alpha = alpha
        self.folding = folding