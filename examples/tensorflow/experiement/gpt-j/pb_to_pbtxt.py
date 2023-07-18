import tensorflow as tf
from tensorflow.python.platform import gfile

def pb_to_pbtxt(input_path, output_path, output_name):
    with gfile.FastGFile(input_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.compat.v1.train.write_graph(graph_def, output_path, output_name, as_text=True)

if __name__ == "__main__":
    pb_to_pbtxt('./converted_graph_def.pb', './', 'converted_graph_def.pbtxt')