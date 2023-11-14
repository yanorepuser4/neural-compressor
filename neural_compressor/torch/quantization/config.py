import torch


class QConfig:
    act_algo = 'minmax'



def get_static_qconfig():
    return QConfig