import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import graph_pb2
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
from tensorflow.python.saved_model import save


def apply_inlining(func):
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

def construct_function_from_graph_def(func, graph_def, frozen_func=None):
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

def get_graph_def_from_func(func):
    inlined_graph_def = apply_inlining(func)
    # self._annotate_variable_ops(func, inlined_graph_def)
    frozen_func = construct_function_from_graph_def(func, inlined_graph_def)

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
    graph_def = tf_optimizer.OptimizeGraph(grappler_session_config,
                                        grappler_meta_graph_def, graph_id=b"tf_graph")
    return graph_def, frozen_func

def parse_saved_model(model, signature_names=[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]):
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1

    if isinstance(model, str):
        # _saved_model = load.load(model, [tag_constants.SERVING])
        _saved_model = load.load(model)
    else:
        _saved_model = model

    graph_func_dict = {}
    for signature_name in signature_names:
        func = _saved_model.signatures[signature_name]
        graph_def, frozen_func = get_graph_def_from_func(func)
        graph_func_dict[signature_name] = [graph_def, func, frozen_func]
    
    return graph_func_dict, _saved_model

def reconstruct_saved_model(graph_func_dict, trackable, path):
    signatures = {}
    for signature_name in graph_func_dict.keys():
        graph_def, func, frozen_func = graph_func_dict[signature_name]
        converted_func = construct_function_from_graph_def(func, graph_def, frozen_func)
        signatures.update({signature_name: converted_func})
    save.save(trackable, path, signatures, options=None)

def get_suffix(input_str):
    """Split the node name into two parts.

    Returns:
        Pure string name without suffix
        Index of the node
    """
    splitted_str = input_str.split(':')
    if len(splitted_str) < 2:
        return input_str, 0

    return splitted_str[0], int(splitted_str[-1])

def node_name_from_input(node_name):
    """Get the original node name from input string.

    Args:
        node_name: input node's name in string

    Returns:
        node's name
    """
    import re
    if node_name.startswith("^"): # pragma: no cover
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name

# def weight_name_mapping(name):
#     name = name.replace('tfgptj_for_causal_lm', 'StatefulPartitionedCall')
#     name = name.replace('kernel:0', 'Tensordot/ReadVariableOp')
#     return name

def weight_name_mapping(name):
    name = 'StatefulPartitionedCall/'+name
    name = name.replace('kernel:0', 'Tensordot/ReadVariableOp')
    return name

# graph_func_dict, _saved_model = parse_saved_model('./gpt-j-6B-2-signatures-first-second-iter', ["serving_default", "serving_first_iteration"])
# # graph_func_dict, _saved_model = parse_saved_model('./gpt-j-6B-2-signatures-first-second-iter')
# reconstruct_saved_model(graph_func_dict, _saved_model, './converted_gpt-j-6B-2-signatures-first-second-iter')
