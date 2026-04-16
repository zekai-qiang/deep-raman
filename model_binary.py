"""
model_binary.py
---------------
Binary 1D-CNN with CBAM attention for Raman spectroscopy classification
(Healthy vs. Myopathy). Includes 5-fold stratified cross-validation and
evaluation metrics (AUROC, precision-recall, confusion matrix).

Usage:
    python model_binary.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    GlobalAveragePooling1D, Dense,
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
)

from data_loading import (
    load_raw_data, preprocess_spectra, group_spectra_by_subject,
    split_by_class, WAVENUMBER_AXIS,
)
from cbam_attention import cbam_block

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FILTERS       = 256
KERNEL_SIZE     = 17
DROPOUT_RATE    = 0.7
L2_REGULARISER  = 1e-4
LEARNING_RATE   = 1e-4
BATCH_SIZE      = 8
N_EPOCHS        = 100
N_FOLDS         = 5
RANDOM_STATE    = 42

# ── Model definition ──────────────────────────────────────────────────────────

def build_binary_cnn_cbam(input_shape):
    """1D-CNN with CBAM for binary classification.

    Parameters
    ----------
    input_shape : tuple  e.g. (901, 1)

    Returns
    -------
    tf.keras.Model  compiled for binary cross-entropy
    """
    input_layer = Input(shape=input_shape)

    x = Conv1D(
        N_FILTERS, KERNEL_SIZE,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(L2_REGULARISER),
    )(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = cbam_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(X, Y):
    """5-fold stratified cross-validation with ROC and PR curve collection.

    Parameters
    ----------
    X : np.ndarray, shape (N, 901, 1)
    Y : np.ndarray, shape (N,)  binary labels {0, 1}

    Returns
    -------
    all_y_true  : list of true labels across all folds
    all_y_pred  : list of predicted probabilities across all folds
    all_history : list of training history dicts
    fprs, tprs  : per-fold ROC curve arrays
    """
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_y_true, all_y_pred, all_history = [], [], []
    fprs, tprs = [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)):
        print(f"Fold {fold + 1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        model = build_binary_cnn_cbam(input_shape=(X.shape[1], 1))
        history = model.fit(
            X_train, y_train,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=2,
        )

        y_pred_prob = model.predict(X_val).flatten()
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred_prob)
        all_history.append(history.history)
        fprs.append(fpr)
        tprs.append(tpr)

        print(classification_report(y_val, (y_pred_prob > 0.5).astype(int)))

    return all_y_true, all_y_pred, all_history, fprs, tprs


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_mean_roc(fprs, tprs):
    common_fpr = np.linspace(0, 1, 100)
    interp_tprs = [np.interp(common_fpr, f, t) for f, t in zip(fprs, tprs)]
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr  = np.std(interp_tprs, axis=0)
    roc_auc  = auc(common_fpr, mean_tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(common_fpr, mean_tpr, color="darkorange", lw=2,
             label=f"Mean AUC = {roc_auc:.2f}")
    plt.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color="gray", alpha=0.2)
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean AUROC (5-Fold CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("binary_roc_curve.png", dpi=150)
    plt.show()


def plot_training_curves(all_history, metric="loss"):
    values     = np.vstack([h[metric]         for h in all_history])
    val_values = np.vstack([h[f"val_{metric}"] for h in all_history])
    epochs     = range(1, values.shape[1] + 1)

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, np.mean(values,     axis=0), label="Training",   color="blue")
    plt.fill_between(epochs,
                     np.mean(values, axis=0) - np.std(values, axis=0),
                     np.mean(values, axis=0) + np.std(values, axis=0),
                     color="blue", alpha=0.2)
    plt.plot(epochs, np.mean(val_values, axis=0), label="Validation", color="red")
    plt.fill_between(epochs,
                     np.mean(val_values, axis=0) - np.std(val_values, axis=0),
                     np.mean(val_values, axis=0) + np.std(val_values, axis=0),
                     color="red", alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Mean Training {metric.capitalize()} (5-Fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"binary_training_{metric}.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred_prob):
    cm = confusion_matrix(y_true, (np.array(y_pred_prob) > 0.5).astype(int))
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Myopathy"],
                yticklabels=["Healthy", "Myopathy"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("binary_confusion_matrix.png", dpi=150)
    plt.show()


def extract_attention_maps(model, X_samples):
    """Extract CBAM multiply-layer outputs for a batch of spectra."""
    attention_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.layers[i].output
            for i in range(len(model.layers))
            if "multiply" in model.layers[i].name
        ],
    )
    return np.array(attention_model.predict(X_samples))


def plot_group_attention_map(model, group_spectra, group_name,
                              cbam_block_index=2):
    """Overlay averaged CBAM attention map on the mean spectrum."""
    attention_maps = extract_attention_maps(model, group_spectra)
    averaged_attention = np.mean(attention_maps, axis=1)[cbam_block_index]
    averaged_spectrum  = np.mean(group_spectra, axis=(0, -1))

    plt.figure(figsize=(6, 4))
    plt.plot(WAVENUMBER_AXIS, averaged_spectrum,
             color="blue", label="Averaged Spectrum", alpha=0.7)
    plt.imshow(
        np.expand_dims(averaged_attention, axis=0),
        cmap="Reds", aspect="auto",
        extent=[WAVENUMBER_AXIS[0], WAVENUMBER_AXIS[-1], 0, 1],
        alpha=0.7,
    )
    plt.title(f"Averaged Attention Map — {group_name}")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"attention_map_{group_name.lower()}.png", dpi=150)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load and preprocess
    raw_spectra, labels_df = load_raw_data()
    corrected_spectra = preprocess_spectra(raw_spectra)
    grouped_array, unique_labels = group_spectra_by_subject(
        corrected_spectra, labels_df
    )
    healthy_array, _, myopathy_array = split_by_class(grouped_array, unique_labels)

    # Build binary dataset (Healthy=0, Myopathy=1)
    X_array = np.concatenate((healthy_array, myopathy_array))
    Y_array = np.concatenate((
        np.repeat(0, len(healthy_array)),
        np.repeat(1, len(myopathy_array)),
    ))

    # Reshape for Conv1D input: (N, 901, 1)
    X_reshaped = X_array.reshape(X_array.shape[0], -1, 1)

    # Cross-validation
    all_y_true, all_y_pred, all_history, fprs, tprs = run_cross_validation(
        X_reshaped, Y_array
    )

    # Evaluation
    print("\nOverall Classification Report:")
    print(classification_report(all_y_true, (np.array(all_y_pred) > 0.5).astype(int)))

    overall_auc = auc(*roc_curve(all_y_true, all_y_pred)[:2])
    print(f"Overall AUROC: {overall_auc:.4f}")

    # Plots
    plot_mean_roc(fprs, tprs)
    plot_training_curves(all_history, metric="loss")
    plot_training_curves(all_history, metric="accuracy")
    plot_confusion_matrix(all_y_true, all_y_pred)