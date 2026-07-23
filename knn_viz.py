import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["toolbar"] = "None"
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import urllib.request
import zipfile
import os

# --- Download data ---
zip_path = "dry_bean.zip"
xlsx_path = "Dry_Bean_Dataset.xlsx"

if not os.path.exists(xlsx_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as zf:
        # find the xlsx inside
        names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        zf.extract(names[0], ".")
        if names[0] != xlsx_path:
            os.rename(names[0], xlsx_path)
    print("Done.")

df = pd.read_excel(xlsx_path)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
print(df["Class"].value_counts().to_string())

# --- Features: Perimeter vs MajorAxisLength ---
X = df[["Perimeter", "MajorAxisLength"]].values
le = LabelEncoder()
y = le.fit_transform(df["Class"])
species_names = le.classes_
n_classes = len(species_names)

COLORS       = ["#E69F00", "#56B4E9", "#009E73", "#000000", "#0072B2", "#D55E00", "#808080"]
REGION_COLORS = ["#E69F0055", "#56B4E955", "#009E7355", "#00000088", "#0072B255", "#D55E0055", "#80808055"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Hyperparameter selection via 5-fold CV ---
k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50]
cv_means, cv_stds = [], []

print(f"\n{'k':>4}  {'cv acc':>8}  {'± std':>7}")
print("-" * 24)
for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, cv=5)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"{k:>4}  {scores.mean():.3f}     ± {scores.std():.3f}")

best_k = k_values[np.argmax(cv_means)]
print(f"\nBest k by CV: {best_k}  (cv acc: {max(cv_means):.3f})")

final_clf = KNeighborsClassifier(n_neighbors=best_k)
final_clf.fit(X_train_s, y_train)
test_acc = final_clf.score(X_test_s, y_test)
print(f"Test accuracy (reported once): {test_acc:.3f}")

# --- Decision boundary grid (in original units, scaled internally) ---
h_x = (X[:, 0].max() - X[:, 0].min()) / 800
h_y = (X[:, 1].max() - X[:, 1].min()) / 800
x_min, x_max = X[:, 0].min() - 500, X[:, 0].max() + 500
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

# --- Precompute predictions for k=1 and best_k ---
clf1 = KNeighborsClassifier(n_neighbors=1).fit(X_train_s, y_train)
clf_best = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_s, y_train)

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_s = scaler.transform(grid_points)
Z1 = clf1.predict(grid_s).reshape(xx.shape)
Zbest = clf_best.predict(grid_s).reshape(xx.shape)

# Find zoom region: densest cluster of data points where the two models disagree
X_all_s = scaler.transform(X)
pred1 = clf1.predict(X_all_s)
pred_best = clf_best.predict(X_all_s)
disagree_pts = X[pred1 != pred_best]

if len(disagree_pts) == 0:
    cx, cy = X[:, 0].mean(), X[:, 1].mean()
else:
    # find densest cluster via KDE peak: use the point with most neighbours within a radius
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=500).fit(disagree_pts)
    log_dens = kde.score_samples(disagree_pts)
    peak_idx = np.argmax(log_dens)
    cx, cy = disagree_pts[peak_idx]

x_span = (x_max - x_min) * 0.08
y_span = (y_max - y_min) * 0.08
zoom_xlim = (cx - x_span, cx + x_span)
zoom_ylim = (cy - y_span, cy + y_span)
print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")

# --- Plot: 2x2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(14, 13))
fig.suptitle("k-NN — Dry Bean Dataset (Perimeter vs. MajorAxisLength)", fontsize=14, fontweight="bold", y=1.0)

region_cmap = ListedColormap(REGION_COLORS[:n_classes])

configs = [
    (axes[0, 0], clf1,    Z1,    (x_min, x_max), (y_min, y_max), f"k=1  (overall)  test acc: {clf1.score(X_test_s, y_test):.2f}"),
    (axes[0, 1], clf_best, Zbest, (x_min, x_max), (y_min, y_max), f"k={best_k}  (overall)  test acc: {test_acc:.2f}"),
    (axes[1, 0], clf1,    Z1,    zoom_xlim,       zoom_ylim,      f"k=1  (zoomed)"),
    (axes[1, 1], clf_best, Zbest, zoom_xlim,       zoom_ylim,      f"k={best_k}  (zoomed)"),
]

for ax, clf, Z, xlim, ylim, title in configs:
    zoomed = xlim == zoom_xlim

    ax.contourf(xx, yy, Z, cmap=region_cmap, levels=np.arange(-0.5, n_classes))
    ax.contour(xx, yy, Z, colors="white", linewidths=0.5,
               levels=np.arange(0.5, n_classes - 1))

    if zoomed:
        # only show incorrectly classified test points
        test_preds = clf.predict(X_test_s)
        incorrect = test_preds != y_test
        for i, (name, color) in enumerate(zip(species_names, COLORS)):
            mask = incorrect & (y_test == i)
            if mask.any():
                ax.scatter(X_test[mask, 0], X_test[mask, 1], c="white", edgecolors=color,
                           linewidths=1.5, s=40, zorder=4, label=name)
    else:
        for i, (name, color) in enumerate(zip(species_names, COLORS)):
            tr = y_train == i
            ax.scatter(X_train[tr, 0], X_train[tr, 1], c=color, label=name,
                       edgecolors="white", linewidths=0.3, s=10, alpha=0.6, zorder=3)
            te = y_test == i
            ax.scatter(X_test[te, 0], X_test[te, 1], c="white", edgecolors=color,
                       linewidths=1.0, s=15, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Perimeter (pixels)")
    ax.set_ylabel("Major Axis Length (pixels)")
    ax.legend(fontsize=6, loc="upper left", markerscale=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Draw zoom box on the overall plots
for ax in (axes[0, 0], axes[0, 1]):
    ax.add_patch(plt.Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        fill=False, edgecolor="black", linewidth=1.5, linestyle="--", zorder=5
    ))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0)
plt.savefig("knn_drybean.png", dpi=600, bbox_inches="tight")
print("Saved knn_drybean.png")
plt.show()
