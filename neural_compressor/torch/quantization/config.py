import torch


class FP8E4M3QConfig:
    dtype=torch.float8_e4m3fn
    act_algo = 'minmax'


class FP8E5M2QConfig:
    dtype=torch.float8_e5m2
    act_algo = 'minmax'


def get_fp8_e4m3_qconfig():
    return FP8E4M3QConfig


def get_fp8_e5m2_qconfig():
    return FP8E5M2QConfig

