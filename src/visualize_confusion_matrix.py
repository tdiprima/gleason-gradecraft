#!/usr/bin/env python3
"""
Script to visualize confusion matrix from CSV file and save as PNG.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the confusion matrix CSV
df = pd.read_csv('confusion_matrix.csv', index_col=0)

# Clean up the column and index names (remove arrows)
df.columns = [col.replace('→', '').strip() for col in df.columns]
df.index = [idx.replace('→', '').strip() for idx in df.index]

# Create figure and axis
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})

# Set labels
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_file = 'confusion_matrix.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to {output_file}")

# Optionally display the plot
# plt.show()
