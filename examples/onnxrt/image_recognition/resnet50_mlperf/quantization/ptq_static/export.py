import argparse
import os

def export_onnx_model(input_model, onnx_model_path):
    os.system('wget https://zenodo.org/record/2535873/files/resnet50_v1.pb -O {}.pb'.format(input_model))
    os.system(f'python -m tf2onnx.convert --input {input_model}.pb --output {onnx_model_path} '
              f'--inputs-as-nchw input_tensor:0 --inputs input_tensor:0 --outputs softmax_tensor:0 --opset 11')
    print("ONNX Model exported to {0}".format(onnx_model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export onnx model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='resnet50_v1',
        help='input model for convert')
    parser.add_argument(
        '--output_model',
        type=str,
        default='resnet50_v1.onnx',
        help='path to exported model file')
    args = parser.parse_args()

    export_onnx_model(args.model_name_or_path, args.output_model)
