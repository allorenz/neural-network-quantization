import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)


COCO_FOLDER = config['coco_folder']
ANNOTATION_FILE_PATH = config['annotation_file_path']
RESULTS_DIR = 'results/'
MODEL_NAME = "ssd_mobilenet"
SSD_MOBILENET_MODEL_DIR = config['model_dir']
MODEL_QUANTIZED_DIR = config["model_quantized_path"]


def evaluate_predictions(results_dir=RESULTS_DIR, model_name=MODEL_NAME):
    annType = 'bbox'
    results_file_path = results_dir + model_name + "_results.json"

    #initialize COCO ground truth api
    cocoGt=COCO(ANNOTATION_FILE_PATH)

    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(results_file_path)

    # only ids that were used for prediction
    with open(results_file_path, 'r') as file:
        results = json.load(file)
    image_ids = [r["image_id"] for r in results]

    # evaluate
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # average precision scores
    #ap_scores = cocoEval.stats
    metrics = generate_metrics_dict(cocoEval)

    return metrics

def generate_metrics_dict(cocoEval):
    keys = [
        "AP_IoU_0.50:0.95_all_maxDets_100",
        "AP_IoU_0.50_all_maxDets_100",
        "AP_IoU_0.75_all_maxDets_100",
        "AP_IoU_0.50:0.95_small_maxDets_100",
        "AP_IoU_0.50:0.95_medium_maxDets_100",
        "AP_IoU_0.50:0.95_large_maxDets_100",
        "AR_IoU_0.50:0.95_all_maxDets_1",
        "AR_IoU_0.50:0.95_all_maxDets_10",
        "AR_IoU_0.50:0.95_all_maxDets_100",
        "AR_IoU_0.50:0.95_small_maxDets_100",
        "AR_IoU_0.50:0.95_medium_maxDets_100",
        "AR_IoU_0.50:0.95_large_maxDets_100"
    ]
    metrics = {key: cocoEval.stats[i] for i, key in enumerate(keys)}
    return metrics

def get_memory_footprint(model_name):
    if model_name == "ssd_mobilenet":
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(SSD_MOBILENET_MODEL_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        return (total_size / (1024 * 1024))
    else:
        model_path = f"{MODEL_QUANTIZED_DIR}/{model_name}.tflite"
        total_size = os.path.getsize(model_path)
        return (total_size / (1024 * 1024))
