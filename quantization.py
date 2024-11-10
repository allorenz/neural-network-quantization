import pathlib
import tensorflow as tf
import json


with open('config.json', 'r') as file:
    config = json.load(file)
model_quantized_path = config["model_quantized_path"]
model_path = config['model_path']


def quantize_models():
    tflite_models_dir = pathlib.Path(model_quantized_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    quantization_config = [tf.float32, tf.float16, "int8"]

    for config in quantization_config:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

        if isinstance(config, tf.dtypes.DType):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [config]
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        model_name = config.name if isinstance(config, tf.dtypes.DType) else config
        tflite_model_file = tflite_models_dir / f"ssd_mobilenet_{model_name}.tflite"
        tflite_model_file.write_bytes(tflite_model)

        print(tflite_model_file)
        print(len(tflite_model))