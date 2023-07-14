import time
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from neural_compressor import METRICS
from neural_compressor.data import TensorflowImageRecord
from neural_compressor.data.dataloaders.tensorflow_dataloader import TensorflowDataLoader
from neural_compressor.data import ComposeTransform
from neural_compressor.data import LabelShift
from neural_compressor.data import TensorflowResizeCropImagenetTransform

from argparse import ArgumentParser
arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--dataset_location', dest='dataset_location',
                          default='/home/zehaohua/TF_Imagenet_SmallData/',
                          help='location of evaluation dataset')
args = arg_parser.parse_args()

def inference(path):
    eval_dataset = TensorflowImageRecord(root=args.dataset_location, transform=ComposeTransform(transform_list= \
            [TensorflowResizeCropImagenetTransform(height=224, width=224, mean_value=[123.68, 116.78, 103.94])]))
    eval_dataloader = TensorflowDataLoader(dataset=eval_dataset, batch_size=1)
    postprocess = LabelShift(label_shift=1)
    metrics = METRICS('tensorflow')
    metric = metrics['topk']()
    model = tf.saved_model.load(path)
    infer = model.signatures["serving_default"]
    output_dict_keys = infer.structured_outputs.keys()
    output_name = list(output_dict_keys )[0]
    def eval_func(dataloader, metric):
        warmup = 5
        iteration = None
        latency_list = []
        iteration = 100
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = np.array(inputs)
            input_tensor = tf.constant(inputs)
            start = time.time()
            predictions = infer(input_tensor)[output_name]
            end = time.time()
            predictions = predictions.numpy()
            predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
            latency_list.append(end - start)
            if iteration and idx >= iteration:
                break
        latency = np.array(latency_list[warmup:]).mean() / eval_dataloader.batch_size
        return latency

    latency = eval_func(eval_dataloader, metric)
    acc = metric.result()
    return latency, acc

if __name__ == "__main__":
    latency1, acc1 = inference('./resnet50')
    latency2, acc2 = inference('./converted_resnet50')
    print('---------------------------------------------------------')
    print('The infrence results of original resnet50 with TF2.x API')
    print("Batch size = {}".format(1))
    print("Latency: {:.3f} ms".format(latency1 * 1000))
    print("Throughput: {:.3f} images/sec".format(1. / latency1))
    print("Accuracy: %.5f" % acc1)
    print('---------------------------------------------------------')
    print('The infrence results of converted resnet50 with TF2.x API')
    print("Batch size = {}".format(1))
    print("Latency: {:.3f} ms".format(latency2 * 1000))
    print("Throughput: {:.3f} images/sec".format(1. / latency2))
    print("Accuracy: %.5f" % acc2)
