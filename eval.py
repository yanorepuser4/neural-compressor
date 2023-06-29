import sys
sys.path.insert(0, './')
import os
os.system("conda activate torch")
from evaluation import evaluate as lm_evaluate
import argparse
import  torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", nargs="?", default="/models/gpt-j-6B"
)
args = parser.parse_args()
model_name = args.model_name
model = torch.load("gpt-j-6b-prune.pt")
device = torch.device("cuda:3")
model.to(device)
results = lm_evaluate(model="hf-causal",
                      model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                      user_model=model, tasks=["lambada_openai"],
                      device=str(device), batch_size=128)