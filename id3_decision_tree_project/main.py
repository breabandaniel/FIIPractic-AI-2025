import os
import pandas as pd

dataset_path = os.path.join("data", "diabetes_dataset.csv")
data = pd.read_csv(dataset_path)


target = "diabet"
train_size = int(0.7 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]