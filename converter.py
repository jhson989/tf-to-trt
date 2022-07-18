import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

INPUT_DIR="./tf_result"
OUTPUT_DIR="./trt_result"
if not os.path.exists(os.path.join(os.getcwd(), OUTPUT_DIR)):
    os.makedirs(os.path.join(os.getcwd(), OUTPUT_DIR))


converter = trt.TrtGraphConverterV2(input_saved_model_dir=INPUT_DIR)
converter.convert()
converter.save(OUTPUT_DIR)