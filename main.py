from data_model_loader import load_model, load_coco_2014_dataset
from quantization import quantize_models
import json
import base_model
import tflite_model
import pathlib


with open('config.json', 'r') as file:
    config = json.load(file)

COCO_FOLDER = config['coco_folder']
MODEL_PATH = config['model_path']
RESULTS_DIR = config["results_dir"]
TFLITE_MODEL_DIR = config["model_quantized_path"]
OUTPUT_DIR = config["output_dir"]

model_selection = ["ssd_mobilenet", "ssd_mobilenet_float32", "ssd_mobilenet_float16", "ssd_mobilenet_int8"]

def main():
    # load model: either from path or from hub
    model = load_model()

    # quantize: tflite converter loads model from path
    quantize_models(MODEL_PATH)

    # load images (names)
    images = load_coco_2014_dataset()

    output = {}

    for model_name in model_selection:
        print(f"Processing {model_name}")

        # predict, store, evaluate: ssd_mobilenet base model 
        if(model_name == "ssd_mobilenet"):
            
            results, inference_times = base_model.predict(model, images, n_images=10, data_path= COCO_FOLDER)
            
            base_model.store_results(results, RESULTS_DIR)
            
            metrics = base_model.evaluate_predictions(RESULTS_DIR, "ssd_mobilenet")
            avg_inference_time = sum(inference_times)/len(inference_times)

            output[model_name] = {
                    "avg_inference_time" : avg_inference_time,
                    "memory_footprint" : 123,
                    "metrics" : metrics
            }
        else:
            
            tflite_model_path = f"{TFLITE_MODEL_DIR}/{model_name}.tflite"# model_name
            
            results, inference_times = tflite_model.predict(tflite_model_path, COCO_FOLDER, images, n_images=10)
            tflite_model.store_results(results, RESULTS_DIR, model_name)
            
            metrics = tflite_model.evaluate_predictions(model_name, RESULTS_DIR)
            avg_inference_time = sum(inference_times)/len(inference_times)
            


            output[model_name] = {
                    "avg_inference_time" : avg_inference_time,
                    "memory_footprint" : 123,
                    "metrics" : metrics
            }


    output_dir = pathlib.Path("output/")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir/"output.json"

    with open(output_file, 'w') as json_file:
            json.dump(output, json_file, indent=4)


if __name__ == '__main__':
    main()