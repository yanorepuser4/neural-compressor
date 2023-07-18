import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import wrap_function
from tensorflow.python.util import nest

from argparse import ArgumentParser
arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--input_model', dest='input_model',
                          help='location of input_model')
args = arg_parser.parse_args()

class DumpSavedModel():
    def _apply_inlining(self, func):
        """Apply an inlining optimization to the function's graph definition."""
        graph_def = func.graph.as_graph_def()

        # In some cases, a secondary implementation of the function (e.g. for GPU) is
        # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
        # TF2 produces a CuDNN-based RNN for GPU).
        # This function suppose to inline all functions calls, but "api_implements"
        # prevents this from happening. Removing the attribute solves the problem.
        # To learn more about "api_implements", see:
        #   tensorflow/core/grappler/optimizers/implementation_selector.h
        for function in graph_def.library.function:
            if "api_implements" in function.attr:
                del function.attr["api_implements"]

        meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)

        # Clear the initializer_name for the variables collections, since they are not
        # needed after saved to saved_model.
        for name in [
            "variables", "model_variables", "trainable_variables", "local_variables"
        ]:
            raw_list = []
            for raw in meta_graph.collection_def["variables"].bytes_list.value:
                variable = variable_pb2.VariableDef()
                variable.ParseFromString(raw)
                variable.ClearField("initializer_name")
                raw_list.append(variable.SerializeToString())
            meta_graph.collection_def[name].bytes_list.value[:] = raw_list

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in func.inputs + func.outputs:
            fetch_collection.node_list.value.append(array.name)
        meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

        # Initialize RewriterConfig with everything disabled except function inlining.
        config = config_pb2.ConfigProto()
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.min_graph_nodes = -1  # do not skip small graphs
        rewrite_options.optimizers.append("function")

        new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)

        return new_graph_def

    def _construct_function_from_graph_def(self, func, graph_def, frozen_func=None):
        """Rebuild function from graph_def."""
        if frozen_func is None:
            frozen_func = func

        # If a function is converted, then the TF context contains the original
        # function while the converted_graph_def contains the converted function.
        # Remove the original function from the TF context in this case.
        for f in graph_def.library.function:
            while context.context().has_function(f.signature.name):
                context.context().remove_function(f.signature.name)

        captures = {
            c[1].name.split(":")[0]: c[0]
            for c in frozen_func.graph.captures
        }
        new_func = wrap_function.function_from_graph_def(
            graph_def, [tensor.name for tensor in frozen_func.inputs],
            [tensor.name for tensor in frozen_func.outputs], captures)
        new_func.graph.structured_outputs = nest.pack_sequence_as(
            func.graph.structured_outputs, new_func.graph.structured_outputs)
        # new_func._function_type = func.function_type  # pylint: disable=protected-access

        # Copy structured input signature from original function (used during
        # serialization)
        new_func.graph.structured_input_signature = (func.structured_input_signature)

        return new_func

    def dump_graph(self, src):
        config = tf.compat.v1.ConfigProto()
        config.use_per_session_threads = 1
        config.inter_op_parallelism_threads = 1

        _saved_model = load.load(src, [tag_constants.SERVING])
        func = _saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        inlined_graph_def = self._apply_inlining(func)
        frozen_func = self._construct_function_from_graph_def(func, inlined_graph_def)

        frozen_graph_def = frozen_func.graph.as_graph_def()
        grappler_meta_graph_def = saver.export_meta_graph(
            graph_def=frozen_graph_def, graph=frozen_func.graph)

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in frozen_func.inputs + frozen_func.outputs:
            fetch_collection.node_list.value.append(array.name)
            grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
                fetch_collection)

        grappler_session_config = config_pb2.ConfigProto()
        rewrite_options = grappler_session_config.graph_options.rewrite_options
        rewrite_options.min_graph_nodes = -1
        extracted_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config,
                                            grappler_meta_graph_def, graph_id=b"tf_graph")

        f=tf.io.gfile.GFile('dumped_graph.pb','wb')
        f.write(extracted_graph_def.SerializeToString()) 

if __name__ == "__main__":
    converter = DumpSavedModel()
    converter.dump_graph(src=args.input_model)