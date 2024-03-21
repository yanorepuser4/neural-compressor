
from neural_compressor.adaptor.torch_utils.auto_round import get_dataloader
from transformers import AutoModel, AutoTokenizer

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
import torch


float_model = AutoModel.from_pretrained(
    "EleutherAI/gpt-neo-125m", torch_dtype=torch.bfloat16)


# from auto_round.calib_dataset import CALIB_DATASETS

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-125m", trust_remote_code=True
)
# get_dataloader = CALIB_DATASETS.get("NeelNanda/pile-10k", CALIB_DATASETS["NeelNanda/pile-10k"])
dataloader = get_dataloader(tokenizer, seqlen=2048)


woq_conf = PostTrainingQuantConfig(approach="weight_only",
                                   op_type_dict={
                                       ".*": {  # re.match
                                            "weight": {
                                                "dtype": "int",
                                                "bits": 4,  # 1-8 bit
                                                "algorithm": "AUTOROUND",
                                            },
                                       }
                                   }
                                   )
quantized_model = fit(model=float_model, conf=woq_conf,
                      calib_dataloader=dataloader)
