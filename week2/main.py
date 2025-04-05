import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.knn import plot_classified_points
from src.knn import plot_misclassified_points
from src.knn import knn_predict

df = pd.read_csv("data/dataset_hipertensiune.csv")
X = df[["IMC", "Colesterol"]].values
y = df["Hipertensiune"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plot_classified_points(X_train, X_test, y_train, y_test,"clasificarea_punctelor.png")
plot_misclassified_points(X_test,y_test, knn_predict(X_train,y_train,X_test,k=5),"clasif-corecta-incorecta.png")


df1 = pd.read_csv("data/dataset_hipertensiune (1).csv")
X1 = df1[["Varsta", "IMC", "Colesterol"]].values
y1 = df1["Hipertensiune"].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler = StandardScaler()
X1_train = scaler.fit_transform(X_train)
X1_test = scaler.transform(X_test)

```plot_classified_points(X1_train, X1_test, y1_train, y1_test,"clasificarea_punctelor1.png")
plot_misclassified_points(X1_test,y1_test, knn_predict(X1_train,y1_train,X1_test,k=5),"clasif-corecta-incorecta1.png")```