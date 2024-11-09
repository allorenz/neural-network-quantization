import tensorflow as tf
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)
coco_folder = config['coco_folder']
annotation_file_path = config['annotation_file_path']

# load image ids
with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)
filename_to_image_id= {image['file_name']: image["id"] for image in coco_data['images']}


def load_tflite_model(tflite_model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def preprocess_image_tflite(image, input_details): 
    if image.mode == 'L':
        image = image.convert('RGB')
    
    # Resize the image to the expected size
    target_shape = input_details[0]['shape'][1:3]  # height, width
    image = image.resize(target_shape)
    
    # Convert to numpy array and scale to [0, 255]
    image_np = np.array(image, dtype=np.uint8)
    
    # Add batch dimension [1, height, width, 3]
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

def run_inference(interpreter, input_details, output_details, image):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output results
    output_data = {}
    for output_detail in output_details:
        output_data[output_detail['name']] = interpreter.get_tensor(output_detail['index'])
    
    return output_data


def predict(tflite_model_path, data_path, images, n_images=5):
    data_path = pathlib.Path(data_path)
    results = []

    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)


    for img in tqdm(images[:n_images], desc="Inferencing images"):
        # load image
        image_path = data_path/"val2014"/"val2014"/img
        image = Image.open(image_path)
        width, height = image.size

        # transform image
        image_np = preprocess_image_tflite(image, input_details)

        # inference
        try:
            output_data = run_inference(interpreter, input_details, output_details, image_np)
        except:
            print(img)
            break 

        num_detections = int(output_data['StatefulPartitionedCall:5'][0])

        results = []

        for i in range(num_detections):
            ymin, xmin, ymax, xmax = output_data['StatefulPartitionedCall:1'][0][i]
            ymin = ymin * height
            ymax = ymax * height
            xmin = xmin * width
            xmax = xmax * width

            result = {
                        "image_id" : filename_to_image_id[img],
                        "category_id": int(output_data['StatefulPartitionedCall:2'][0][i]),
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], 
                        "score": float(output_data['StatefulPartitionedCall:4'][0][i])
            }
            results.append(result)
    return results