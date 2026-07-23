import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

le = LabelEncoder()
y = le.fit_transform(df["Class"])
features = [c for c in df.columns if c != "Class"]
X_all = df[features].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

k = 10  # fixed k for comparison
results = []

for f1, f2 in combinations(range(len(features)), 2):
    # scale so units don't bias distance
    scaler = StandardScaler()
    X_pair_train = scaler.fit_transform(X_train[:, [f1, f2]])
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_pair_train, y_train, cv=5)
    results.append((scores.mean(), features[f1], features[f2]))

results.sort(reverse=True)

print(f"{'cv acc':>8}  feature pair")
print("-" * 50)
for acc, f1, f2 in results[:20]:
    print(f"  {acc:.3f}   {f1} vs {f2}")
