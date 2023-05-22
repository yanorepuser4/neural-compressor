"""Tests for quantization"""
import numpy as np
import unittest
import os
import shutil


def build_fake_model():
    import tensorflow as tf
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), name='y', dtype=tf.float32)
            op = tf.nn.conv2d(input=x, filter=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), name='y', dtype=tf.float32)
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

class TestTpeStrategy(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    def test_run_tpe_one_trial(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
        from neural_compressor.data import Datasets, DATALOADERS

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy='tpe', max_trials=200)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)
        def eval_func(model):
            return 1
        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tune_cri, accuracy_criterion=acc_cri)
        q_model = fit(model=self.constant_graph,
                      conf=conf,
                      calib_dataloader=dataloader,
                      eval_func=eval_func)

    def test_run_tpe_max_trials(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
        from neural_compressor.data import Datasets, DATALOADERS

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy='tpe', max_trials=5)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)

        from neural_compressor.metric import METRICS
        metrics = METRICS('tensorflow')
        top1 = metrics['topk']()
        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tune_cri, accuracy_criterion=acc_cri)
        q_model = fit(model=self.constant_graph,
                      conf=conf,
                      calib_dataloader=dataloader,
                      eval_dataloader=dataloader,
                      eval_metric=top1)

if __name__ == "__main__":
    unittest.main()
