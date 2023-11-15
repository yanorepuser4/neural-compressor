import torch


float8_e4m3 = torch.float8_e4m3fn
float8_e5m2 = torch.float8_e5m2

# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240*0.9, dtype=torch.float).to('hpu')
E5M2_AMAX = torch.tensor(57344*0.9, dtype=torch.float).to('hpu')