# onnxruntime quantization API:
# from onnxruntime.quantization import quantize_dynamic, quantize_static
from neural_compressor.onnxruntime.utility import (
    get_demo_model, 
    get_dummy_dataloader,
    get_default_quant_config,
    get_default_tuning_config,
    get_default_quant_with_tuning_config
    )
from neural_compressor.onnxruntime.quantization import quantize

fp32_model = get_demo_model()
quant_config = get_default_quant_config(),


# # not tuning
# fp32_model = get_demo_model()
# dummy_dataloader = get_dummy_dataloader()
# q_model = quantize(
#     fp32_model,
#     calib_dataloader=dummy_dataloader,
#     quant_config=quant_config)


# tuning
tuning_config = get_default_tuning_config()
quant_config = get_default_quant_with_tuning_config()
dummy_dataloader = get_dummy_dataloader()
q_model = quantize(
    fp32_model,
    calib_dataloader=dummy_dataloader,
    quant_config = quant_config,
    tuning_config = tuning_config)
