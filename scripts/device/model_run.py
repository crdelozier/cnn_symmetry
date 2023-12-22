import sys
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from optparse import OptionParser
import time

parser = OptionParser()
parser.add_option("--base_model_file", type=str)
parser.add_option("-t", "--change_type",
                  action="store_false", dest="change_type", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

if not options.base_model_file:
  print("Must specify base model file")
  sys.exit(1)

base_model_file = options.base_model_file

# Load the TensorFlow Lite model.
base_model = tf.lite.Interpreter(model_path=base_model_file)

# Allocate memory for the model.
base_model.allocate_tensors()

input_details = base_model.get_input_details()[0]
input_shape = input_details["shape"]
input_type = input_details["dtype"]
print(input_shape)
print(input_type)

output_details = base_model.get_output_details()[0]
output_shape = output_details["shape"]
output_type = output_details["dtype"]

# TODO: Strip input layers for "transfer learning"
input_tensor = tf.random.uniform(input_shape,dtype=input_type)

# Time and run model
base_model.set_tensor(input_details["index"], input_tensor)

invoke_time_start = time.perf_counter_ns()
base_model.invoke()
invoke_time = time.perf_counter_ns() - invoke_time_start

print(str(invoke_time))
