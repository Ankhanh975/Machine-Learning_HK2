from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_estimator(name: str, value: float | int):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=value, max_iter=500, random_state=42),
        )
    if name == "rbf_svc":
        return SVC(C=value, kernel="rbf", gamma=0.001)
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=value)
    raise ValueError(f"Unknown estimator: {name}")


def spatial_average_features(X: np.ndarray, grid_size: int) -> np.ndarray:
    """Return one average-intensity feature per cell in a regular image grid."""
    images = X.reshape(-1, 28, 28)
    block_size = 28 // grid_size
    pooled = images.reshape(
        -1, grid_size, block_size, grid_size, block_size
    ).mean(axis=(2, 4))
    return pooled.reshape(len(X), -1)


def add_coarse_spatial_features(X: np.ndarray) -> np.ndarray:
    coarse_grid = spatial_average_features(X, grid_size=7)
    return np.concatenate((X, coarse_grid), axis=1)


def add_spatial_and_shape_features(X: np.ndarray) -> np.ndarray:
    images = X.reshape(-1, 28, 28)
    fine_grid = spatial_average_features(X, grid_size=14)
    vertical_symmetry = -np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
    horizontal_symmetry = -np.abs(images - images[:, ::-1, :]).mean(axis=(1, 2))
    row_intensity = images.mean(axis=2)
    column_intensity = images.mean(axis=1)
    shape_features = np.column_stack(
        (vertical_symmetry, horizontal_symmetry, row_intensity, column_intensity)
    )
    return np.concatenate((X, fine_grid, shape_features), axis=1)


def evaluate_cv(
    estimator_name: str,
    parameter_value: float | int,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    fold_accuracies = []
    for train_idx, validation_idx in folds:
        model = make_estimator(estimator_name, parameter_value)
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[validation_idx])
        fold_accuracies.append(
            accuracy_score(y[validation_idx], predictions)
        )
    return fold_accuracies


print("Loading MNIST...")
X, y = fetch_openml(
    "mnist_784", version=1, as_frame=False, return_X_y=True
)
print(f"Loaded {len(X)} examples with {X.shape[1]} features.")

# Use the first 10,000 training examples and the provided final 10,000 test examples.
X_train = np.asarray(X[:10_000], dtype=np.float32) / 255.0
y_train = np.asarray(y[:10_000])
X_test = np.asarray(X[60_000:], dtype=np.float32) / 255.0
y_test = np.asarray(y[60_000:])
print(f"Using {len(X_train)} for cross-validation and {len(X_test)} for test.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train))

# Question 6.1: compare three sklearn estimators, each with three settings.
search_space = {
    "logistic_regression": [0.1, 1.0, 10.0],  # C
    "rbf_svc": [1.0, 5.0, 10.0],  # C; gamma is fixed at 0.001
    "gaussian_nb": [1e-9, 1e-8, 1e-7],  # var_smoothing
}
estimator_results = []

print("\nQuestion 6.1: estimator and hyperparameter cross-validation")
for estimator_name, values in search_space.items():
    for value in values:
        scores = evaluate_cv(
            estimator_name, value, X_train, y_train, folds
        )
        result = {
            "estimator": estimator_name,
            "value": value,
            "fold_accuracies": scores,
            "mean_cv_accuracy": float(np.mean(scores)),
        }
        estimator_results.append(result)
        print(
            f"{estimator_name:22s} value={value!s:>8s} "
            f"folds={np.round(scores, 4).tolist()} "
            f"mean={result['mean_cv_accuracy']:.4f}"
        )

best_result = max(
    estimator_results, key=lambda result: result["mean_cv_accuracy"]
)
print(
    "\nBest estimator from 6.1: "
    f"{best_result['estimator']} "
    f"(parameter={best_result['value']}, "
    f"mean CV accuracy={best_result['mean_cv_accuracy']:.4f})"
)

# Question 6.2: compare the same estimator and hyperparameter using the same folds.
# Both transformations append hand-crafted features to the original 784 pixels.
representations = {
    "original_pixels": X_train,
    "pixels_plus_7x7_intensity_grid": add_coarse_spatial_features(X_train),
    "pixels_plus_14x14_grid_symmetry_projections": (
        add_spatial_and_shape_features(X_train)
    ),
}

representation_results = []
print("\nQuestion 6.2: feature representation comparison")
for representation_name, features in representations.items():
    scores = evaluate_cv(
        best_result["estimator"],
        best_result["value"],
        features,
        y_train,
        folds,
    )
    result = {
        "representation": representation_name,
        "dimensions": features.shape[1],
        "fold_accuracies": scores,
        "mean_cv_accuracy": float(np.mean(scores)),
    }
    representation_results.append(result)
    print(
        f"{representation_name:48s} "
        f"dimensions={result['dimensions']:4d} "
        f"folds={np.round(scores, 4).tolist()} "
        f"mean={result['mean_cv_accuracy']:.4f}"
    )

best_representation = max(
    representation_results, key=lambda result: result["mean_cv_accuracy"]
)
raw_result = representation_results[0]
best_engineered_result = max(
    representation_results[1:], key=lambda result: result["mean_cv_accuracy"]
)
improvement = (
    best_engineered_result["mean_cv_accuracy"]
    - raw_result["mean_cv_accuracy"]
)
print(
    "\nSelected representation by mean CV accuracy: "
    f"{best_representation['representation']} "
    f"({best_representation['dimensions']} features)"
)
print(
    f"Best engineered representation vs. raw pixels: "
    f"{improvement * 100:+.2f} percentage points."
)
if improvement > 1e-12:
    print("The engineered features improved mean CV accuracy for this model.")
elif improvement < -1e-12:
    print(
        "The engineered features reduced mean CV accuracy for this model; "
        "they may add redundant or noisy information."
    )
else:
    print(
        "The best engineered features matched raw-pixel mean CV accuracy; "
        "they did not provide a measurable improvement."
    )
print(
    "Feature rationale: the 7x7 grid summarizes where ink is concentrated; "
    "the 14x14 grid preserves finer spatial layout, while the two symmetry "
    "scores and row/column projections describe shape and stroke distribution."
)

# Keep the test set untouched until the estimator and representation are selected.
if best_representation["representation"] == "original_pixels":
    X_train_final = X_train
    X_test_final = X_test
elif best_representation["representation"] == "pixels_plus_7x7_intensity_grid":
    X_train_final = add_coarse_spatial_features(X_train)
    X_test_final = add_coarse_spatial_features(X_test)
else:
    X_train_final = add_spatial_and_shape_features(X_train)
    X_test_final = add_spatial_and_shape_features(X_test)

final_model = make_estimator(best_result["estimator"], best_result["value"])
final_model.fit(X_train_final, y_train)
test_accuracy = accuracy_score(y_test, final_model.predict(X_test_final))

print("\nFinal selected model")
print(f"Estimator: {best_result['estimator']}")
print(f"Hyperparameter value: {best_result['value']}")
print(f"Representation: {best_representation['representation']}")
print(f"Feature dimensions: {best_representation['dimensions']}")
print(f"Mean cross-validation accuracy: {best_representation['mean_cv_accuracy']:.4f}")
print(f"Test accuracy (evaluated once): {test_accuracy:.4f}")
