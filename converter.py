from tensorflow.python.compiler.tensorrt import trt_convert as trt

INPUT_DIR="./result"
OUTPUT_DIR="./trt"

converter = trt.TrtGraphConverterV2(input_saved_model_dir=INPUT_DIR)
converter.convert()
converter.save(OUTPUT_DIR)