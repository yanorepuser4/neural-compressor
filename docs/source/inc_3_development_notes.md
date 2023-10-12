# Interface between Strategy and adaptor

### TODO list

- Strategy
    - [x] Base sampler
    - [ ] Dependency sampler 
- [] Objective
    - Handle general stop
    - Handle the O0 stop
- [] Metric

### Change
- Removed `query_capability`
    - The adaptor registers its own `sampler`
- `eval_metric` and `eval_dataloader`
    - if user pass `eval_metric` and `eval_dataloader` instead of `eval_func`, adaptor needs to construct the `eval_func` corresponding and pass it to the strategy

### Open
- change `strategy` into `tuner`?


### Demo code for both INC developer ans end user

- How do INC developers register a new sampler.

```python
# how does the INC developer register the new sampler.
# should we define a separate sampler class for each stages or use a generic sampler class with different name.

sq_alpha = [0.1, 0.2, 0.3]
sq_hp = HyperParams(name="sq_alpha", params_space={"alpha": sq_alpha})
strategy.add_sampler(BaseSampler(sq_hp, name="sq_alpha", priority=1))

```

- How does the end user override the sampler order.
```python
# user pass the priority of sampler.
# how does the user know which sampler are supported by adaptor.

# we's better support the import from INC instead of ask user to write sampler name 

from neural_compressor.onnxruntime.config import SmoothQuantSamplerConfig, OpTypeWiseSamplerConfig

SmoothQuantSamplerConfig.priority = 100
OpTypeWiseSamplerConfig.priority = 1

SmoothQuantSamplerConfig.alpha_lst = [0.1, 0.2, 0.3]

tuning_config = TuningConfig()


```
- Arch
```
----------------------------
End User           : 
----------------------------
Adaptor Developer  : the adaptor merge the end user config and pass it into strategy(register sampler)
----------------------------
Strategy Developer : the strategy not handle the end user's config directly
----------------------------
``