import time
import statistics
import torch
import matplotlib.pyplot as plt

from torchvision import models
from PIL import Image
from tkinter import Tk, filedialog


# ==================================================
# Choose an image
# ==================================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Choose an image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.webp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if image_path == "":
    print("No image selected.")
    exit()

image = Image.open(image_path).convert("RGB")

print("\nAI BENCHMARK RESULTS")
print("=" * 35)


# ==================================================
# Store results
# ==================================================

results = {}


# ==================================================
# Benchmark function
# ==================================================

def benchmark_model(model_name, model, weights, device_name):
    device = torch.device(device_name)

    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    # Warm-up
    with torch.no_grad():
        model(image_tensor)

    if device_name == "mps":
        torch.mps.synchronize()

    runs = 100
    run_times = []

    for i in range(runs):
        start = time.perf_counter()

        with torch.no_grad():
            output = model(image_tensor)

        if device_name == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()

        run_times.append(
            (end - start) * 1000
        )

    average_time = statistics.mean(run_times)
    minimum_time = min(run_times)
    maximum_time = max(run_times)
    standard_deviation = statistics.stdev(run_times)

    # Prediction
    probabilities = torch.softmax(
        output[0],
        dim=0
    )

    top_probability, top_id = torch.max(
        probabilities,
        dim=0
    )

    categories = weights.meta["categories"]

    prediction_name = categories[
        top_id.item()
    ]

    confidence = (
        top_probability.item() * 100
    )

    return {
        "average": average_time,
        "minimum": minimum_time,
        "maximum": maximum_time,
        "std_dev": standard_deviation,
        "prediction": prediction_name,
        "confidence": confidence,
        "parameters": parameter_count
    }


# ==================================================
# MobileNetV2
# ==================================================

mobilenet_weights = models.MobileNet_V2_Weights.DEFAULT

results["MobileNetV2"] = {}

mobilenet_cpu = models.mobilenet_v2(
    weights=mobilenet_weights
)

results["MobileNetV2"]["CPU"] = benchmark_model(
    "MobileNetV2",
    mobilenet_cpu,
    mobilenet_weights,
    "cpu"
)

if torch.backends.mps.is_available():

    mobilenet_gpu = models.mobilenet_v2(
        weights=mobilenet_weights
    )

    results["MobileNetV2"]["GPU"] = benchmark_model(
        "MobileNetV2",
        mobilenet_gpu,
        mobilenet_weights,
        "mps"
    )


# ==================================================
# ResNet18
# ==================================================

resnet_weights = models.ResNet18_Weights.DEFAULT

results["ResNet18"] = {}

resnet_cpu = models.resnet18(
    weights=resnet_weights
)

results["ResNet18"]["CPU"] = benchmark_model(
    "ResNet18",
    resnet_cpu,
    resnet_weights,
    "cpu"
)

if torch.backends.mps.is_available():

    resnet_gpu = models.resnet18(
        weights=resnet_weights
    )

    results["ResNet18"]["GPU"] = benchmark_model(
        "ResNet18",
        resnet_gpu,
        resnet_weights,
        "mps"
    )


# ==================================================
# Print clean summary
# ==================================================

for model_name in results:

    print(f"\n{model_name}")

    cpu = results[model_name]["CPU"]

    print(
        f"CPU: {cpu['average']:.2f} ms"
    )

    if "GPU" in results[model_name]:

        gpu = results[model_name]["GPU"]

        print(
            f"GPU: {gpu['average']:.2f} ms"
        )

        speedup = (
            cpu["average"]
            / gpu["average"]
        )

        print(
            f"GPU speedup: {speedup:.2f}x"
        )

    print(
        f"Prediction: {cpu['prediction']} "
        f"({cpu['confidence']:.2f}%)"
    )

    print("-" * 35)


# ==================================================
# Find fastest result
# ==================================================

fastest_model = None
fastest_device = None
fastest_time = float("inf")

for model_name, devices in results.items():

    for device_name, data in devices.items():

        if data["average"] < fastest_time:

            fastest_time = data["average"]
            fastest_model = model_name
            fastest_device = device_name


print("\nFASTEST RESULT")

print(
    f"{fastest_model} on {fastest_device}: "
    f"{fastest_time:.2f} ms"
)


# ==================================================
# Create grouped graph
# ==================================================

model_names = list(results.keys())

cpu_times = [
    results[name]["CPU"]["average"]
    for name in model_names
]

gpu_times = [
    results[name]["GPU"]["average"]
    for name in model_names
]

x = list(range(len(model_names)))

width = 0.35

plt.figure(figsize=(9, 6))

cpu_bars = plt.bar(
    [i - width / 2 for i in x],
    cpu_times,
    width,
    label="CPU"
)

gpu_bars = plt.bar(
    [i + width / 2 for i in x],
    gpu_times,
    width,
    label="GPU"
)

plt.title(
    "AI Inference Performance: CPU vs GPU",
    fontsize=16
)

plt.xlabel("Model")

plt.ylabel(
    "Average Inference Time (ms)"
)

plt.xticks(
    x,
    model_names
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)


# Add time labels above bars
for bar in cpu_bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        height + 0.5,

        f"{height:.1f} ms",

        ha="center"
    )


for bar in gpu_bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        height + 0.5,

        f"{height:.1f} ms",

        ha="center"
    )


# Add speedup labels
for i, model_name in enumerate(model_names):

    cpu = results[model_name]["CPU"]["average"]

    gpu = results[model_name]["GPU"]["average"]

    speedup = cpu / gpu

    plt.text(
        i,
        max(cpu, gpu) + 3,

        f"{speedup:.2f}x faster",

        ha="center"
    )


plt.tight_layout()

plt.savefig(
    "cpu_vs_gpu_benchmark.png",
    dpi=300
)

plt.show()