import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Sample data (replace this with your actual data loading)
with open("speech_result.json", "r") as fh:
    data = json.load(fh)

# Extract predictions and labels
predictions = [item['generated'] for item in data['generated_texts']]
labels = [item['label'] for item in data['generated_texts']]

# Define sorted classes with specific order
priority_classes = ["neutral", "sustain", "change"]
other_classes = sorted(list(set(labels + predictions) - set(priority_classes)))
classes = priority_classes + other_classes

# Generate the confusion matrix
cm = confusion_matrix(labels, predictions, labels=classes)

# Plotting the confusion matrix
plt.figure(figsize=(16, 14))  # Bigger figure size for clarity
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=classes, yticklabels=classes,
                 linewidths=1.5, linecolor='black', annot_kws={"size": 28})

# Change font color on axis labels only
for tick_label in ax.get_xticklabels():
    if tick_label.get_text() in priority_classes:
        tick_label.set_color('#A50021')  # Crimson Red
    else:
        tick_label.set_color('#1F4E79')  # Dark Blue

for tick_label in ax.get_yticklabels():
    if tick_label.get_text() in priority_classes:
        tick_label.set_color('#A50021')  # Crimson Red
    else:
        tick_label.set_color('#1F4E79')  # Dark Blue


plt.xlabel('Predicted Labels', fontsize=26)
plt.ylabel('True Labels', fontsize=26)
plt.xticks(fontsize=24, rotation=45)
plt.yticks(fontsize=24, rotation=0)
plt.title('Confusion Matrix', fontsize=30)
plt.tight_layout()

plt.savefig('cm.png')  # Save the figure for LaTeX inclusion
plt.show()