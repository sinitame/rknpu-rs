import sys
import tensorflow as tf
from rknn.api import RKNN


def get_model_infos(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    input_infos = interpreter.get_input_details()[0]
    output_infos = interpreter.get_output_details()[0]
    return (input_infos, output_infos)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        model_path = "/home/model.tflite"
        output_path = "/home/model.rknn"

    (input_infos, output_infos) = get_model_infos(model_path)

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print("--> Config model")
    rknn.config(target_platform="rk3588")
    print("done")

    print("--> Loading model")
    ret = rknn.load_tflite(model=model_path)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")
