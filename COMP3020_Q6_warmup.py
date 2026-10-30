from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# MNIST setup from the assignment
# ------------------------------------------------------------
print("Loading MNIST...")
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
print(f"Loaded {len(X)} examples with {X.shape[1]} features.")

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
X_train, y_train = (X_train[:10000], y_train[:10000])  # reduce dataset size
print(f"Using {len(X_train)} for cross-validation and {len(X_test)} for test.")

# ------------------------------------------------------------
# Manual 5-fold cross-validation without GridSearchCV
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidate_estimators = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=500, multi_class="auto", random_state=42),
        "params": {"C": [0.1, 1.0, 10.0]},
    },
    "naive_bayes": {
        "estimator": GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [10, 20, 30]},
    },
}

results = []

for estimator_name, config in candidate_estimators.items():
    base_estimator = config["estimator"]
    param_grid = config["params"]

    for param_name, param_values in param_grid.items():
        fold_accuracies = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            model = clone(base_estimator)
            model.set_params(**{param_name: param_values[0]})

            # Use a small scaling pipeline for logistic regression, because the pixels have
            # a different scale than the learned parameters in the raw image space.
            if estimator_name == "logistic_regression":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", model),
                ])

            # Build the correct parameter values for each candidate hyperparameter.
            # Only one parameter is tuned at a time for this implementation, which matches
            # the assignment requirement to tune at least 3 estimators and at least one
            # hyperparameter each.
            if param_name == "C":
                model.set_params(clf__C=param_values[0])
            elif param_name == "var_smoothing":
                model.set_params(var_smoothing=param_values[0])
            elif param_name == "max_depth":
                model.set_params(max_depth=param_values[0])

            # The above is a bit cumbersome because some models are not pipeline-wrapped.
            # To keep the logic transparent and fully manual, we rebuild an estimator for each
            # hyperparameter value from the scratch below.
            if estimator_name == "logistic_regression":
                for current_value in param_values:
                    fold_accuracies_for_value = []
                    for current_train_idx, current_val_idx in cv.split(X_train, y_train):
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(C=current_value, max_iter=500, multi_class="auto", random_state=42)),
                        ])
                        model.fit(X_train[current_train_idx], y_train[current_train_idx])
                        val_pred = model.predict(X_train[current_val_idx])
                        fold_accuracies_for_value.append(accuracy_score(y_train[current_val_idx], val_pred))

                    mean_acc = float(np.mean(fold_accuracies_for_value))
                    results.append({
                        "estimator": estimator_name,
                        "param_name": param_name,
                        "param_value": current_value,
                        "fold_accuracies": fold_accuracies_for_value,
                        "mean_cv_accuracy": mean_acc,
                    })
                    print(f"{estimator_name:20s} {param_name:15s}={current_value:>8}  mean_cv_accuracy={mean_acc:.4f}")
                break

            elif estimator_name == "naive_bayes":
                for current_value in param_values:
                    fold_accuracies_for_value = []
                    for current_train_idx, current_val_idx in cv.split(X_train, y_train):
                        model = GaussianNB(var_smoothing=current_value)
                        model.fit(X_train[current_train_idx], y_train[current_train_idx])
                        val_pred = model.predict(X_train[current_val_idx])
                        fold_accuracies_for_value.append(accuracy_score(y_train[current_val_idx], val_pred))

                    mean_acc = float(np.mean(fold_accuracies_for_value))
                    results.append({
                        "estimator": estimator_name,
                        "param_name": param_name,
                        "param_value": current_value,
                        "fold_accuracies": fold_accuracies_for_value,
                        "mean_cv_accuracy": mean_acc,
                    })
                    print(f"{estimator_name:20s} {param_name:15s}={current_value:>8}  mean_cv_accuracy={mean_acc:.4f}")
                break

            elif estimator_name == "decision_tree":
                for current_value in param_values:
                    fold_accuracies_for_value = []
                    for current_train_idx, current_val_idx in cv.split(X_train, y_train):
                        model = DecisionTreeClassifier(max_depth=current_value, random_state=42)
                        model.fit(X_train[current_train_idx], y_train[current_train_idx])
                        val_pred = model.predict(X_train[current_val_idx])
                        fold_accuracies_for_value.append(accuracy_score(y_train[current_val_idx], val_pred))

                    mean_acc = float(np.mean(fold_accuracies_for_value))
                    results.append({
                        "estimator": estimator_name,
                        "param_name": param_name,
                        "param_value": current_value,
                        "fold_accuracies": fold_accuracies_for_value,
                        "mean_cv_accuracy": mean_acc,
                    })
                    print(f"{estimator_name:20s} {param_name:15s}={current_value:>8}  mean_cv_accuracy={mean_acc:.4f}")
                break

# ------------------------------------------------------------
# Select the best estimator/hyperparameter pair by mean CV accuracy
# ------------------------------------------------------------
best_result = max(results, key=lambda x: x["mean_cv_accuracy"])
print("\nBest hyperparameter configuration by mean CV accuracy:")
print(best_result)

if best_result["estimator"] == "logistic_regression":
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_result["param_value"], max_iter=500, multi_class="auto", random_state=42)),
    ])
elif best_result["estimator"] == "naive_bayes":
    final_model = GaussianNB(var_smoothing=best_result["param_value"])
elif best_result["estimator"] == "decision_tree":
    final_model = DecisionTreeClassifier(max_depth=best_result["param_value"], random_state=42)
else:
    raise ValueError("Unknown selected estimator.")

final_model.fit(X_train, y_train)
test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test accuracy of the selected model: {test_accuracy:.4f}")

# ------------------------------------------------------------
# Optional: a more structured summary for the assignment report
# ------------------------------------------------------------
print("\nSummary for writing up the assignment:")
print(f"- Estimator: {best_result['estimator']}")
print(f"- Tuned hyperparameter: {best_result['param_name']} = {best_result['param_value']}")
print(f"- Fold accuracies: {best_result['fold_accuracies']}")
print(f"- Mean CV accuracy: {best_result['mean_cv_accuracy']:.4f}")
print(f"- Test accuracy: {test_accuracy:.4f}")
