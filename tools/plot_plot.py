import json
import matplotlib.pyplot as plt
import os

# Define the path template and noise levels
path_template = "/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/test_speech_epoch_100_step_160000_classification_noise{}/test_results.json"
noise_levels = [25, 20, 15, 10, 5] #, 0]

# Collect accuracy values
accuracies = []

for noise in noise_levels:
    file_path = path_template.format(noise)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            accuracies.append(data.get("accuracy", 0))
    else:
        print(f"File not found for noise level {noise}: {file_path}")
        accuracies.append(None)  # Handle missing files



# Plotting
plt.figure(figsize=(8, 6))
plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Noise Level')
plt.xticks(noise_levels)
plt.xlim(27, 3)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_noise.png")
