"""
model_multiclass_scaled.py
--------------------------
Multi-class 1D-CNN with CBAM for Raman spectroscopy classification using
a larger nested cross-validation scheme (15 outer / 10 inner folds) and
explicit temperature scaling for calibrated softmax probabilities.

This is the production-scale variant of model_multiclass.py, optimised
for datasets with more subjects.

Usage:
    python model_multiclass_scaled.py
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
INPUT_SHAPE         = (4, 901)
N_FILTERS           = 128
KERNEL_SIZE         = 11
DROPOUT_RATE        = 0.7
L2_REGULARISER      = 1e-4
LEARNING_RATE       = 1e-4
BATCH_SIZE          = 8
N_EPOCHS_INNER      = 20
N_EPOCHS_RETRAIN    = 20
N_OUTER_FOLDS       = 15
N_INNER_FOLDS       = 10
TEMP_SCALING_STEPS  = 100
TEMP_SCALING_LR     = 1e-4
RANDOM_STATE        = 42


def build_model(input_shape, n_classes):
    """1D-CNN + CBAM returning raw logits (softmax applied post temperature-scaling)."""
    input_layer = Input(shape=input_shape)
    x = Conv1D(N_FILTERS, KERNEL_SIZE, padding="same", activation="relu",
               kernel_regularizer=regularizers.L2(L2_REGULARISER))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = cbam_block(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    logits = Dense(n_classes)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def fit_temperature(logits, y_one_hot,
                    n_steps=TEMP_SCALING_STEPS, lr=TEMP_SCALING_LR):
    """Optimise scalar temperature to minimise cross-entropy of calibrated probabilities."""
    temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    optimizer   = tf.keras.optimizers.Adam(learning_rate=lr)
    y_tensor    = tf.convert_to_tensor(np.array(y_one_hot), dtype=tf.float32)

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    y_tensor, tf.math.divide(logits, temperature)
                )
            )
        grads = tape.gradient(loss, temperature)
        optimizer.apply_gradients([(grads, temperature)])

    return temperature


def scale_batch(X, scaler):
    """Apply MinMax scaling preserving original array shape."""
    return scaler.fit_transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)


def run_nested_cv(X_array, Y_array):
    Y_one_hot = tf.one_hot(Y_array, N_CLASSES)
    scaler    = MinMaxScaler()

    outer_skf = StratifiedKFold(N_OUTER_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    inner_skf = StratifiedKFold(N_INNER_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)

    all_true, all_pred, all_conf      = [], [], []
    all_attention, all_spectra        = [], []
    best_model, best_acc              = None, 0.0

    for o_idx, (o_train, o_test) in enumerate(outer_skf.split(X_array, Y_array)):
        print(f"\nOuter Fold {o_idx + 1}/{N_OUTER_FOLDS}")

        Xo_train = tf.gather(X_array,  o_train).numpy()
        Xo_test  = tf.gather(X_array,  o_test).numpy()
        Yo_train = tf.gather(Y_one_hot, o_train)
        Yo_test  = tf.gather(Y_one_hot, o_test)

        Xo_test_sc = scale_batch(Xo_test, scaler)
        model      = build_model(INPUT_SHAPE, N_CLASSES)

        for i_idx, (i_train, i_val) in enumerate(
                inner_skf.split(Xo_train, np.argmax(Yo_train, axis=1))):
            print(f"  Inner Fold {i_idx + 1}/{N_INNER_FOLDS}")
            Xi_train = scale_batch(Xo_train[i_train], scaler)
            Xi_val   = scale_batch(Xo_train[i_val],   scaler)
            Yi_train = tf.gather(Yo_train, i_train)
            Yi_val   = tf.gather(Yo_train, i_val)
            model.fit(Xi_train, Yi_train,
                      epochs=N_EPOCHS_INNER, batch_size=BATCH_SIZE,
                      validation_data=(Xi_val, Yi_val), verbose=2)

        Xo_train_sc  = scale_batch(Xo_train, scaler)
        logits_train = model.predict(Xo_train_sc)
        temperature  = fit_temperature(logits_train, Yo_train)

        logits_test   = model.predict(Xo_test_sc)
        scaled_logits = tf.math.divide(logits_test, temperature)
        pred_probs    = tf.nn.softmax(scaled_logits).numpy()
        pred_labels   = np.argmax(pred_probs, axis=1)
        true_labels   = np.argmax(Yo_test, axis=1)

        fold_acc = accuracy_score(true_labels, pred_labels)
        if fold_acc > best_acc:
            best_acc   = fold_acc
            best_model = model

        attn_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.layers[i].output
                     for i in range(len(model.layers))
                     if "multiply" in model.layers[i].name],
        )
        attn_maps = np.array(attn_model.predict(Xo_test_sc[..., :2]))
        all_attention.append(np.mean(attn_maps, axis=1))
        all_spectra.append(Xo_test_sc)

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        all_conf.extend(pred_probs)

        print(f"\nOuter Fold {o_idx + 1} Report:")
        print(classification_report(true_labels, pred_labels,
                                     target_names=CLASS_NAMES))

    all_attention = np.array(list(chain(all_attention)))
    all_spectra   = np.array(list(chain(all_spectra)))

    print("\nOverall Classification Report:")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES))

    return all_true, all_pred, all_conf, all_attention, all_spectra, best_model


def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("scaled_confusion_matrix.png", dpi=150)
    plt.show()


def plot_attention_maps(attention_scores, class_arrays, true_labels):
    true_arr = np.array(true_labels)
    for cls_idx, cls_arr in enumerate(class_arrays):
        mask = np.where(true_arr == cls_idx)
        avg_attn = np.mean(attention_scores[mask], axis=0)
        avg_spec = np.mean(np.mean(cls_arr, axis=1), axis=0)
        plt.figure(figsize=(6, 4))
        plt.plot(WAVENUMBER_AXIS, avg_spec, color="blue",
                 label="Averaged Spectrum", alpha=0.7)
        plt.imshow(np.expand_dims(avg_attn, axis=0), cmap="Reds",
                   aspect="auto", extent=[900, 1800, 0, 140], alpha=0.7)
        plt.title(f"Attention Map — {CLASS_NAMES[cls_idx]}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"scaled_attn_{CLASS_NAMES[cls_idx].lower()}.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    raw_spectra, labels_df       = load_raw_data()
    corrected_spectra            = preprocess_spectra(raw_spectra)
    grouped_array, unique_labels = group_spectra_by_subject(
        corrected_spectra, labels_df
    )
    healthy_arr, neuropathy_arr, myopathy_arr = split_by_class(
        grouped_array, unique_labels
    )
    X_array, Y_array = build_dataset(
        healthy_arr, neuropathy_arr, myopathy_arr, unique_labels
    )

    true_labels, pred_labels, pred_conf, attn, spectra, best_model = \
        run_nested_cv(X_array, Y_array)

    plot_confusion_matrix(true_labels, pred_labels)
    plot_attention_maps(attn, [healthy_arr, neuropathy_arr, myopathy_arr],
                        true_labels)

    results_df = pd.DataFrame({
        "True_Label":        np.array(true_labels),
        "Predicted_Label":   np.array(pred_labels),
        "Confidence_Class0": np.array(pred_conf)[:, 0],
        "Confidence_Class1": np.array(pred_conf)[:, 1],
        "Confidence_Class2": np.array(pred_conf)[:, 2],
    })
    results_df.to_csv("scaled_results.csv", index=False)
    print(results_df.to_string())

    # Optional: retrain best model on full dataset and save
    # scaler = MinMaxScaler()
    # X_full_sc = scaler.fit_transform(
    #     X_array.reshape(-1, X_array.shape[-1])
    # ).reshape(X_array.shape)
    # best_model.fit(X_full_sc, tf.one_hot(Y_array, N_CLASSES),
    #                epochs=N_EPOCHS_RETRAIN, batch_size=BATCH_SIZE, verbose=2)
    # best_model.save("best_scaled_model.keras")