import os
import shutil
import unittest
import torch
import torchvision
import onnx
from neural_compressor.data import Datasets, DATALOADERS

def export_onnx_cv_model(model, path, opset=12):
    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    path,                      # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=opset,          # the ONNX version to export the model to, please ensure at least 11.
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ["input"],   # the model"s input names
                    output_names = ["output"], # the model"s output names
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                                  "output" : {0 : "batch_size"}})


class TestAdaptorONNXRT(unittest.TestCase):

    rn50_export_path = "rn50.onnx"
    rn50_model = torchvision.models.resnet18()
    datasets = Datasets('onnxrt_qlinearops')
    cv_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
    cv_dataloader = DATALOADERS['onnxrt_qlinearops'](cv_dataset)


    @classmethod
    def setUpClass(self):
        export_onnx_cv_model(self.rn50_model, self.rn50_export_path, 12)
        export_onnx_cv_model(self.rn50_model, 'rn50_9.onnx', 9)
        self.rn50_model = onnx.load(self.rn50_export_path)
        
        
    def test_auto_quant_v2(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
        tuning_criterion = TuningCriterion(max_trials=8, timeout=10000)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=-0.01)
        conf = PostTrainingQuantConfig(quant_level=1, approach="auto",
                                       op_type_dict={"Add|MatMul|Conv": {'weight': {'algorithm': ['minmax']},\
                                           'activation': {'algorithm': ['minmax']}}},
                                       tuning_criterion=tuning_criterion, 
                                       accuracy_criterion=accuracy_criterion)
        conf.framework = "onnxrt_qlinearops"
        q_model = fit(model=self.rn50_model, conf=conf, calib_dataloader=self.cv_dataloader, eval_func=lambda model: 1)
        self.assertIsNotNone(q_model)
        
if __name__ == "__main__":
    unittest.main()
