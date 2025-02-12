## Motivation

As AI models grow larger and more powerful, computational resource consumption also increases, leading to higher energy usage and greater carbon emissions. Various techniques, such as model pruning, distillation, and quantization, can reduce memory footprint and computational requirements during inference; however, these methods often come at the cost of model accuracy. This project focuses on post-training quantizing model weights to evaluate the trade-off between memory footprint reduction, inference time and accuracy loss.

## Methodology

Quantization reduces the precision of a model's weights by converting them from higher-bit representations (e.g., float32) to lower-bit formats (e.g., float16 or int8), which decreases the model's memory usage and computational requirements. However, this reduction in precision can lead to a slight loss in accuracy, as the lower-bit representations may not capture the full range of values as accurately as higher-precision formats.

We use the TFLiteConverter to convert TensorFlow models into TensorFlow Lite (TFLite) format, modifying the model weights from float32 to float16 and int8 representations. Finally, we evaluate the models' accuracy to assess the impact of these changes on performance.

## Results

### Object Detection Model

#### Inference Time and Memory Consumption

| Model                   | Memory Footprint (MB) | Mean Inference Time (s) |
|--------------------------|----------------------:|------------------------:|
| ssd_mobilenet_float32     |               23.720  |                0.115962 |
| ssd_mobilenet_float16     |               12.166  |                0.117698 |
| ssd_mobilenet_int8        |                6.707  |                0.167178 |

The float32 model has the largest memory footprint (23.72 MB) and the fastest inference time (0.1160s). The float16 model nearly halves the memory usage (12.17 MB) with minimal impact on speed (0.1177s). The int8 model reduces memory by 3.5x (6.71 MB) but increases inference time to 0.1672s.

#### Average Precision & Average Recall

![avg precision plot](object_detection_model/output/average_precision_plot.png)

![avg precision plot](object_detection_model/output/average_recall_plot.png)

The plot demonstrates that precision remains consistent between the float32 and float16 models, while the int8 model experiences a significant decline in performance. Across all image sizes, the int8 model shows consistently unsatisfactory results, indicating a noticeable loss in precision.