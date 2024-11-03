
import os
from ultralytics import YOLO
import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights
import logging

model_path = r"D:\SWTNK-YOLO\runs\train\ours(SWTNK-YOLO)\weights\best.pt"  # Path to the trained model
img_folder = r'D:\SWTNK-YOLO\result'  # Folder containing images
save_folder = r'D:\SWTNK-YOLO\result1'  # Folder where the results will be saved

# Create the directory to save results if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Load the model
model = YOLO(model_path)

# Iterate through all images in the folder
for img_name in os.listdir(img_folder):
    if not img_name.endswith(".jpg"):
        continue

    # Get full path to the image
    img_path = os.path.join(img_folder, img_name)

    # Perform prediction
    results = model.predict(img_path)

    # Generate a full save path
    save_path = os.path.join(save_folder, img_name)  # Save image with the same name in the save folder

    # Save the detection result
    results[0].plot(save=True, filename=save_path)  # Use 'filename' argument to specify the save location

    print(f"Saved results for {img_name} in {save_path}")

import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_weight_size(path):
    try:
        stats = os.stat(path)
        return f'{stats.st_size / (1024 ** 2):.1f}'
    except OSError as e:
        logging.error(f"Error getting weight size: {e}")
        return "N/A"


def warmup_model(model, device, example_inputs, iterations=200):
    logging.info("Beginning warmup...")
    for _ in tqdm(range(iterations), desc='Warmup'):
        model(example_inputs)


def test_model_latency(model, device, example_inputs, iterations=1000):
    logging.info("Testing latency...")
    time_arr = []
    for _ in tqdm(range(iterations), desc='Latency Test'):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start_time = time.time()

        model(example_inputs)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end_time = time.time()
        time_arr.append(end_time - start_time)

    return np.mean(time_arr), np.std(time_arr)


def main(opt):
    device = select_device(opt.device)
    weights = opt.weights
    assert weights.endswith('.pt'), "Model weights must be a .pt file."

    model = attempt_load_weights(weights, device=device, fuse=True)
    model = model.to(device)
    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)

    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()

    warmup_model(model, device, example_inputs, opt.warmup)
    mean_latency, std_latency = test_model_latency(model, device, example_inputs, opt.testtime)

    logging.info(f"Model weights: {opt.weights} Size: {get_weight_size(opt.weights)}M "
                 f"(Batch size: {opt.batch}) Latency: {mean_latency:.5f}s ± {std_latency:.5f}s "
                 f"FPS: {1 / mean_latency:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test YOLOv10 model performance.")
    parser.add_argument('--weights', type=str, default=r'D:\SWTNK-YOLO\runs\train\ours(SWTNK-YOLO)\weights\best.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='image sizes [height, width]')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='warmup iterations')
    parser.add_argument('--testtime', default=1000, type=int, help='test iterations for latency')
    parser.add_argument('--half', action='store_true', help='use FP16 mode for inference')
    opt = parser.parse_args()

    main(opt)

import os
import pandas as pd
import matplotlib.pyplot as plt


# Helper function to smooth the data using a moving average
def moving_average(data, window_size=5):
    return data.rolling(window=window_size).mean()


# Updated function to handle plotting with smoothing and checking for anomalies
def plot_metrics_and_loss(experiment_names, metrics_info, loss_info, metrics_subplot_layout, loss_subplot_layout,
                          metrics_figure_size=(15, 10), loss_figure_size=(15, 10), base_directory='runs/train'):
    # Plot metrics
    plt.figure(figsize=metrics_figure_size)
    for i, (metric_name, title) in enumerate(metrics_info):
        plt.subplot(*metrics_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')  # Correctly joining the path
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue  # Skip this file if not found
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            smoothed_data = moving_average(data[column_name])  # Apply smoothing
            plt.plot(smoothed_data, label=name)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    metrics_filename = 'metrics_curves.png'
    plt.savefig(metrics_filename)
    plt.show()

    # Plot loss
    plt.figure(figsize=loss_figure_size)
    for i, (loss_name, title) in enumerate(loss_info):
        plt.subplot(*loss_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')  # Correctly joining the path
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue  # Skip this file if not found
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == loss_name][0]
            smoothed_data = moving_average(data[column_name])  # Apply smoothing
            plt.plot(smoothed_data, label=name)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    loss_filename = 'loss_curves.png'
    plt.savefig(loss_filename)
    plt.show()

    return metrics_filename, loss_filename


# Metrics to plot
metrics_info = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
    ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
]

# Loss to plot
loss_info = [
    ('train/box_loss', 'Training Box Loss'),
    ('train/cls_loss', 'Training Classification Loss'),
    ('train/dfl_loss', 'Training DFL Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/cls_loss', 'Validation Classification Loss'),
    ('val/dfl_loss', 'Validation DFL Loss')
]

# Plot the metrics and loss from multiple experiments
metrics_filename, loss_filename = plot_metrics_and_loss(
    experiment_names=['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x','ours(SWTNK-YOLO)'],
    metrics_info=metrics_info,
    loss_info=loss_info,
    metrics_subplot_layout=(2, 2),
    loss_subplot_layout=(2, 3)
)

print(f"Saved plots: {metrics_filename}, {loss_filename}")