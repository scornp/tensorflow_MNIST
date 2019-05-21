import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.ERROR)


#stem = './Result_02_IOAdded/IOAdded'
meta_path = './Result_02_IOAdded/IOAdded.meta' # Your .meta file
output_pb = './Result_03_frozen/frozen.pb'
output_node_names = ['output']    # Output nodes

if not os.path.exists('./Result_03_frozen'):
    os.makedirs('./Result_03_frozen')

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('./Result_02_IOAdded'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open(output_pb, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

