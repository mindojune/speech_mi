import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the path template and noise levels
# path_template = "/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/test_speech_epoch_100_step_160000_classification_noise{}/test_results.json"
path_template = "/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/test_speech_epoch_4_step_8000_forecasting_noise{}/test_results.json"

noise_levels = [25, 20, 15, 10, 5, 0]

# Number of bootstrap samples
num_bootstrap_samples = 1000

# Collect accuracy values with bootstrap
def bootstrap_confidence_interval(data, num_samples=1000, ci=95):
    bootstrap_samples = np.random.choice(data, (num_samples, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

accuracies = []
lower_bounds = []
upper_bounds = []

for noise in noise_levels:
    file_path = path_template.format(noise)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            accuracy = data.get("accuracy", 0)
            # Simulate data for bootstrapping (assuming binary correct/incorrect predictions)
            simulated_results = [1] * int(accuracy * 1000) + [0] * int((1 - accuracy) * 1000)
            lower, upper = bootstrap_confidence_interval(simulated_results, num_bootstrap_samples)
            
            accuracies.append(accuracy)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
    else:
        print(f"File not found for noise level {noise}: {file_path}")
        accuracies.append(None)
        lower_bounds.append(None)
        upper_bounds.append(None)

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(noise_levels, accuracies, yerr=[
    np.array(accuracies) - np.array(lower_bounds), 
    np.array(upper_bounds) - np.array(accuracies)
], fmt='o-', color='b', ecolor='orange', elinewidth=2, capsize=5, label='Accuracy with 95% CI')

plt.xlabel('SNR in db', fontsize=26)
plt.ylabel('Accuracy', fontsize=26)
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
plt.title('Model Accuracy vs. Noise Level', fontsize=26)
plt.xticks(noise_levels)
plt.ylim(0.65,0.75)
plt.xlim(27, -2)  # Set x-axis to start from 25 and end at 0
plt.grid(True)
plt.legend(fontsize=26)
plt.tight_layout()
plt.savefig("snr_forecasting.png")
