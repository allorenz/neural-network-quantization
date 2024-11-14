import json
import tensorflow as tf
import base_model
import tflite_model
import pathlib
from evaluate import evaluate_predictions, get_memory_footprint, create_plots
from data_model_loader import load_model, load_coco_2014_dataset
from quantization import quantize_models

with open('config.json', 'r') as file:
    config = json.load(file)

COCO_FOLDER = config['coco_folder']
MODEL_PATH = config['model_path']
RESULTS_DIR = config["results_dir"]
TFLITE_MODEL_DIR = config["model_quantized_path"]
OUTPUT_DIR = config["output_dir"]

model_selection = ["ssd_mobilenet", "ssd_mobilenet_float32", "ssd_mobilenet_float16", "ssd_mobilenet_int8"]

def main():
    # download eval images, returns image names
    image_names = load_coco_2014_dataset()

    # load model: either from path or from hub
    model = load_model()

    # quantize: tflite converter loads model from path
    quantize_models(MODEL_PATH)

    output = {}

    for model_name in model_selection:       
        # predict, store, evaluate: ssd_mobilenet base model 
        if(model_name == "ssd_mobilenet"): 
            results, inference_times = base_model.predict(
                 model=model, 
                 image_names=image_names, 
                 n_images=100, 
                 data_path= COCO_FOLDER
            )
            base_model.store_results(results, RESULTS_DIR)
            
            metrics = evaluate_predictions(results_dir=RESULTS_DIR, model_name=model_name)
            avg_inference_time = sum(inference_times)/len(inference_times)
            memory_footprint = get_memory_footprint(model_name)

            output[model_name] = {
                    "avg_inference_time" : avg_inference_time,
                    "memory_footprint" : memory_footprint,
                    "metrics" : metrics
            }
        else:
            # load tf lite model
            tflite_model_path = f"{TFLITE_MODEL_DIR}/{model_name}.tflite"# model_name
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            
            results, inference_times = tflite_model.predict(
                 interpreter= interpreter, 
                 image_names=image_names, 
                 n_images=100, 
                 data_path= COCO_FOLDER
            )
            tflite_model.store_results(results, RESULTS_DIR, model_name)
            
            metrics = evaluate_predictions(RESULTS_DIR, model_name)
            avg_inference_time = sum(inference_times)/len(inference_times)
            memory_footprint = get_memory_footprint(model_name)
            
            output[model_name] = {
                    "avg_inference_time" : avg_inference_time,
                    "memory_footprint" : memory_footprint,
                    "metrics" : metrics
            }

    output_dir = pathlib.Path("output/")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir/"output.json"

    with open(output_file, 'w') as json_file:
            json.dump(output, json_file, indent=4)

    create_plots()


if __name__ == '__main__':
    main()