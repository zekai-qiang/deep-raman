"""
model_multiclass.py
-------------------
Multi-class 1D-CNN with CBAM for Raman spectroscopy classification
(Healthy / Neuropathy / Myopathy).

Features
--------
- Nested stratified cross-validation (outer 5-fold, inner 5-fold)
- Per-fold MinMax feature scaling
- Softmax output with temperature scaling for probability calibration
- CBAM attention map extraction and group-level visualisation
- Confusion matrix and classification report

Usage:
    python model_multiclass.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    GlobalMaxPooling1D, Dense,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from data_loading import (
    load_raw_data, preprocess_spectra, group_spectra_by_subject,
    split_by_class, build_dataset, WAVENUMBER_AXIS, CLASS_NAMES,
)
from cbam_attention import cbam_block

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_CLASSES           = 3
INPUT_SHAPE         = (4, 901)       # (replicates per subject, wavenumbers)
N_FILTERS           = 128
KERNEL_SIZE         = 11
DROPOUT_RATE        = 0.7
L2_REGULARISER      = 1e-4
LEARNING_RATE       = 1e-4
BATCH_SIZE          = 8
N_EPOCHS_INNER      = 10
N_EPOCHS_RETRAIN    = 20
N_OUTER_FOLDS       = 5
N_INNER_FOLDS       = 5
TEMP_SCALING_STEPS  = 100
TEMP_SCALING_LR     = 1e-4
RANDOM_STATE        = 42


# ── Model definition ──────────────────────────────────────────────────────────

def build_multiclass_cnn_cbam(input_shape, n_classes):
    """1D-CNN with CBAM block for multi-class softmax classification.

    Parameters
    ----------
    input_shape : tuple  e.g. (4, 901)
    n_classes   : int

    Returns
    -------
    tf.keras.Model  (outputs raw logits for temperature scaling compatibility)
    """
    input_layer = Input(shape=input_shape)

    x = Conv1D(
        N_FILTERS, KERNEL_SIZE,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.L2(L2_REGULARISER),
    )(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = cbam_block(x)

    x = GlobalMaxPooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    logits = Dense(n_classes)(x)          # raw logits — no activation

    model = tf.keras.models.Model(inputs=input_layer, outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# ── Temperature scaling ───────────────────────────────────────────────────────

def fit_temperature(logits, y_one_hot, n_steps=TEMP_SCALING_STEPS,
                    lr=TEMP_SCALING_LR):
    """Learn a scalar temperature T to calibrate softmax probabilities.

    Minimises cross-entropy of softmax(logits / T) against ground-truth labels.

    Parameters
    ----------
    logits    : np.ndarray, shape (N, n_classes)
    y_one_hot : tf.Tensor or np.ndarray, shape (N, n_classes)

    Returns
    -------
    temperature : tf.Variable (scalar float)
    """
    temperature = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    optimizer   = tf.keras.optimizers.Adam(learning_rate=lr)

    y_tensor = tf.convert_to_tensor(y_one_hot, dtype=tf.float32)

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            scaled_logits = tf.math.divide(logits, temperature)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y_tensor, scaled_logits)
            )
        grads = tape.gradient(loss, temperature)
        optimizer.apply_gradients([(grads, temperature)])

    return temperature


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_nested_cross_validation(X_array, Y_array):
    """Nested stratified cross-validation with temperature-scaled predictions.

    Returns
    -------
    all_true_labels  : list
    all_pred_labels  : list
    all_pred_conf    : list of softmax probability vectors
    all_attention_scores : np.ndarray
    all_spectra      : np.ndarray
    best_model       : tf.keras.Model  (highest outer-fold accuracy)
    """
    Y_one_hot = tf.one_hot(Y_array, N_CLASSES)
    scaler    = MinMaxScaler()

    outer_skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)

    all_true_labels, all_pred_labels, all_pred_conf = [], [], []
    all_attention_scores, all_spectra = [], []
    best_model, best_accuracy = None, 0.0

    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(
            outer_skf.split(X_array, Y_array)):
        print(f"\nOuter Fold {outer_idx + 1}")

        X_outer_train = tf.gather(X_array,  outer_train_idx)
        X_outer_test  = tf.gather(X_array,  outer_test_idx)
        Y_outer_train = tf.gather(Y_one_hot, outer_train_idx)
        Y_outer_test  = tf.gather(Y_one_hot, outer_test_idx)

        shape = X_outer_test.shape
        X_outer_test_scaled = scaler.fit_transform(
            np.reshape(X_outer_test, (-1, shape[-1]))
        ).reshape(shape)

        model = build_multiclass_cnn_cbam(INPUT_SHAPE, N_CLASSES)

        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(
                    X_outer_train,
                    np.argmax(Y_outer_train, axis=1)
                )):
            print(f"  Inner Fold {inner_idx + 1}")
            X_inner_train = tf.gather(X_outer_train, inner_train_idx)
            X_inner_val   = tf.gather(X_outer_train, inner_val_idx)
            Y_inner_train = tf.gather(Y_outer_train, inner_train_idx)
            Y_inner_val   = tf.gather(Y_outer_train, inner_val_idx)

            shape_train = X_inner_train.shape
            shape_val   = X_inner_val.shape

            X_inner_train_sc = scaler.fit_transform(
                np.reshape(X_inner_train, (-1, shape_train[-1]))
            ).reshape(shape_train)
            X_inner_val_sc = scaler.fit_transform(
                np.reshape(X_inner_val, (-1, shape_val[-1]))
            ).reshape(shape_val)

            model.fit(
                X_inner_train_sc, Y_inner_train,
                epochs=N_EPOCHS_INNER,
                batch_size=BATCH_SIZE,
                validation_data=(X_inner_val_sc, Y_inner_val),
                verbose=2,
            )

        # Temperature scaling on outer training set
        shape_full = X_outer_train.shape
        X_outer_train_sc = scaler.fit_transform(
            np.reshape(X_outer_train, (-1, shape_full[-1]))
        ).reshape(shape_full)

        logits_train = model.predict(X_outer_train_sc)
        temperature  = fit_temperature(logits_train, Y_outer_train)

        # Outer test evaluation
        logits_test    = model.predict(X_outer_test_scaled)
        scaled_logits  = tf.math.divide(logits_test, temperature)
        pred_probs     = tf.nn.softmax(scaled_logits)
        pred_labels    = np.argmax(pred_probs, axis=1)
        true_labels    = np.argmax(Y_outer_test, axis=1)

        fold_accuracy = accuracy_score(true_labels, pred_labels)
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_model    = model

        # Attention maps
        attention_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.layers[i].output
                for i in range(len(model.layers))
                if "multiply" in model.layers[i].name
            ],
        )
        attention_maps = np.array(
            attention_model.predict(X_outer_test_scaled[..., :2])
        )
        attention_scores = np.mean(attention_maps, axis=1)

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        all_pred_conf.extend(pred_probs.numpy())
        all_attention_scores.append(attention_scores)
        all_spectra.append(X_outer_test_scaled)

        print(f"\nOuter Fold Classification Report:")
        print(classification_report(true_labels, pred_labels,
                                     target_names=CLASS_NAMES))

    all_attention_scores = np.array(list(chain(all_attention_scores)))
    all_spectra          = np.array(list(chain(all_spectra)))

    print("\nOverall Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels,
                                 target_names=CLASS_NAMES))

    return (all_true_labels, all_pred_labels, all_pred_conf,
            all_attention_scores, all_spectra, best_model)


# ── Final retraining ──────────────────────────────────────────────────────────

def retrain_best_model(best_model, X_array, Y_one_hot):
    """Fine-tune the best model on the complete dataset."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(
        np.reshape(X_array, (-1, X_array.shape[-1]))
    ).reshape(X_array.shape)

    best_model.fit(
        X_scaled, Y_one_hot,
        epochs=N_EPOCHS_RETRAIN,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    return best_model, scaler


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("multiclass_confusion_matrix.png", dpi=150)
    plt.show()


def plot_group_attention_maps(attention_scores, spectra, true_labels,
                               class_arrays):
    """Visualise group-level averaged attention overlaid on mean spectra."""
    true_labels_arr = np.array(true_labels)
    class_tags      = ["healthy", "neuropathy", "myopathy"]

    for class_idx, (tag, class_arr) in enumerate(zip(class_tags, class_arrays)):
        mask                = np.where(true_labels_arr == class_idx)
        averaged_attention  = np.mean(attention_scores[mask], axis=0)
        averaged_spectrum   = np.mean(np.mean(class_arr, axis=1), axis=0)

        plt.figure(figsize=(6, 4))
        plt.plot(WAVENUMBER_AXIS, averaged_spectrum,
                 color="blue", label="Averaged Spectrum", alpha=0.7)
        plt.imshow(
            np.expand_dims(averaged_attention, axis=0),
            cmap="Reds", aspect="auto",
            extent=[WAVENUMBER_AXIS[0], WAVENUMBER_AXIS[-1], 0, 140],
            alpha=0.7,
        )
        plt.title(f"Group-level Attention Map — {CLASS_NAMES[class_idx]}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"attention_map_{tag}.png", dpi=150)
        plt.show()


# ── Results table ─────────────────────────────────────────────────────────────

def build_results_dataframe(true_labels, pred_labels, pred_conf):
    conf_arr = np.array(pred_conf)
    return pd.DataFrame({
        "True_Label":        np.array(true_labels),
        "Predicted_Label":   np.array(pred_labels),
        "Confidence_Class0": conf_arr[:, 0],
        "Confidence_Class1": conf_arr[:, 1],
        "Confidence_Class2": conf_arr[:, 2],
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_spectra, labels_df   = load_raw_data()
    corrected_spectra        = preprocess_spectra(raw_spectra)
    grouped_array, unique_labels = group_spectra_by_subject(
        corrected_spectra, labels_df
    )
    healthy_arr, neuropathy_arr, myopathy_arr = split_by_class(
        grouped_array, unique_labels
    )
    X_array, Y_array = build_dataset(
        healthy_arr, neuropathy_arr, myopathy_arr, unique_labels
    )

    (true_labels, pred_labels, pred_conf,
     attention_scores, spectra, best_model) = run_nested_cross_validation(
        X_array, Y_array
    )

    plot_confusion_matrix(true_labels, pred_labels)
    plot_group_attention_maps(
        attention_scores, spectra, true_labels,
        [healthy_arr, neuropathy_arr, myopathy_arr],
    )

    results_df = build_results_dataframe(true_labels, pred_labels, pred_conf)
    print(results_df.to_string())
    results_df.to_csv("multiclass_results.csv", index=False)

    # Optional: save best model
    # best_model.save("best_multiclass_model.keras")