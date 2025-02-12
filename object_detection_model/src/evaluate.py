import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# load file paths
with open('config.json', 'r') as file:
    config = json.load(file)


COCO_FOLDER = config['coco_folder']
ANNOTATION_FILE_PATH = config['annotation_file_path']
RESULTS_DIR = '../results/'
MODEL_NAME = "ssd_mobilenet"
SSD_MOBILENET_MODEL_DIR = config['model_dir']
MODEL_QUANTIZED_DIR = config["model_quantized_path"]
OUTPUT_FILE_PATH = config["output_file_path"]


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
    

def create_plots():
    with open(OUTPUT_FILE_PATH, 'r') as f:
        data = json.load(f)

    ap_model_names = []
    ap_metric_names = []
    ap_values = []

    ar_model_names = []
    ar_metric_names = []
    ar_values = []

    metrics = [
        "AP_IoU_0.50:0.95_all_maxDets_100",
        "AP_IoU_0.50:0.95_small_maxDets_100",
        "AP_IoU_0.50:0.95_medium_maxDets_100",
        "AP_IoU_0.50:0.95_large_maxDets_100",
        "AR_IoU_0.50:0.95_all_maxDets_100",
        "AR_IoU_0.50:0.95_small_maxDets_100",
        "AR_IoU_0.50:0.95_medium_maxDets_100",
        "AR_IoU_0.50:0.95_large_maxDets_100"
    ]

    for model, model_data in data.items():
        for metric, value in model_data['metrics'].items():
            # Filter metrics starting with 'AP'
            if metric.startswith('AP') and metric in metrics:
                ap_model_names.append(model)
                ap_metric_names.append(metric)
                ap_values.append(value)
            elif metric.startswith('AR') and metric in metrics:
                ar_model_names.append(model)
                ar_metric_names.append(metric)
                ar_values.append(value)


    df_ap = pd.DataFrame({
        "Model" : ap_model_names,
        "Metric" : ap_metric_names,
        "Value" : ap_values
    })

    df_ar = pd.DataFrame({
        "Model" : ar_model_names,
        "Metric" : ar_metric_names,
        "Value" : ar_values
    })

    # AP
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_ap, x='Model', y='Value', hue='Metric')
    plt.xticks(rotation=0)
    plt.title('Average Precision (AP)')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title="Metric", loc="upper right", fontsize=8.8, handlelength=1)
    plt.tight_layout()
    plt.savefig('../output/average_precision_plot.png', format='png', dpi=300)
    plt.show()

    # AR
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_ar, x='Model', y='Value', hue='Metric')
    plt.xticks(rotation=0)
    plt.title('Average Precision (AR)')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title="Metric", loc="upper right", fontsize=8.8, handlelength=1)
    plt.tight_layout()
    plt.savefig('../output/average_recall_plot.png', format='png', dpi=300)
    plt.show()
