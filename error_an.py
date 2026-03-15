import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("outputs_cv/error_by_length_bin.csv")

# Create bar chart
plt.figure(figsize=(7,4))
plt.bar(df["bin"], df["mean"])

# Labels
plt.xlabel("Sentence length (words)")
plt.ylabel("Mean absolute prediction error (ms)")
plt.title("Prediction Error by Sentence Length")

# Rotate x labels for readability
plt.xticks(rotation=30)

# Layout
plt.tight_layout()

# Save figure
plt.savefig("outputs_cv/error_by_length_plot.png", dpi=300)

plt.show()