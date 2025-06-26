
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === SETTINGS ===
data_dir = "data/"
participant_file = [f for f in os.listdir(data_dir) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(data_dir, participant_file))

# === PLOT RESPONSE RATES ===
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='frequency', hue='heard')
plt.title('Detected Pitch by Frequency')
plt.ylabel('Trial Count')
plt.xlabel('Frequency (Hz)')
plt.legend(title='Heard Pitch')
plt.tight_layout()
plt.savefig('output/detection_plot.png')
plt.show()

# === CONFIDENCE RATINGS ===
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['heard'] == 'y'], x='frequency', y='confidence')
plt.title('Confidence Ratings for Detected Pitches')
plt.ylabel('Confidence (1–5)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('output/confidence_boxplot.png')
plt.show()
