"""
model_vae.py
------------
Variational autoencoder suite for unsupervised representation learning
on Raman spectra. Provides:

    - VAE     : standard variational autoencoder (β = 0)
    - BetaVAE : disentangled VAE with configurable β
    - CVAE    : conditional VAE with class-label conditioning

Includes downstream latent-space classifiers and visualisation utilities.

Usage:
    python model_vae.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
import umap

from data_loading import (
    load_raw_data, preprocess_spectra, WAVENUMBER_AXIS, CLASS_NAMES,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM          = 2
N_CLASSES           = 3
LEARNING_RATE       = 1e-4
BATCH_SIZE          = 32
N_EPOCHS_VAE        = 50
N_EPOCHS_BETA_VAE   = 10
BETA                = 4.0        # β for β-VAE (set 0 for standard VAE)
N_FOLDS_DOWNSTREAM  = 5
RANDOM_STATE        = 42


# ── Sampling layer ────────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """Reparameterisation trick: z = μ + ε·σ, ε ~ N(0, I)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch   = tf.shape(z_mean)[0]
        dim     = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ── Encoder / Decoder builders ────────────────────────────────────────────────

def build_dense_encoder(input_dim, latent_dim):
    """Fully-connected encoder: input → latent Gaussian parameters."""
    encoder_inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(512, activation="relu")(encoder_inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="sigmoid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="sigmoid")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_mean    = layers.BatchNormalization()(z_mean)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = layers.BatchNormalization()(z_log_var)

    z = Sampling()([z_mean, z_log_var])

    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def build_dense_decoder(latent_dim, output_dim):
    """Fully-connected decoder: latent vector → reconstructed spectrum."""
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(128, activation="sigmoid")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="sigmoid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    output = layers.Dense(output_dim, activation="sigmoid")(x)

    return keras.Model(latent_inputs, output, name="decoder")


def build_conv_encoder(input_dim, latent_dim):
    """1D-convolutional encoder for temporal Raman spectra."""
    encoder_inputs = keras.Input(shape=(input_dim, 1))

    x = layers.Conv1D(512, 5, activation="relu", padding="same")(encoder_inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_mean    = layers.BatchNormalization()(z_mean)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = layers.BatchNormalization()(z_log_var)

    z = Sampling()([z_mean, z_log_var])

    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="conv_encoder")


# ── VAE ───────────────────────────────────────────────────────────────────────

class VAE(keras.Model):
    """Standard Variational Autoencoder (β = 0 equivalent)."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker         = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker            = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction      = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=-1,
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ── β-VAE ─────────────────────────────────────────────────────────────────────

class BetaVAE(VAE):
    """Disentangled β-VAE. Set beta=0 to recover standard VAE behaviour."""

    def __init__(self, encoder, decoder, beta=4.0, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.beta = beta

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction      = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=-1,
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ── CVAE ──────────────────────────────────────────────────────────────────────

class CVAE(tf.keras.Model):
    """Conditional VAE: encoder and decoder both receive the class label."""

    def __init__(self, input_dim, condition_dim, latent_dim, batch_size):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_input  = tf.keras.layers.Input(shape=(input_dim,))
        condition_input = tf.keras.layers.Input(shape=(condition_dim,))
        x = tf.keras.layers.Concatenate(axis=-1)([encoder_input, condition_input])
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        z_mean    = tf.keras.layers.Dense(latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = tf.keras.Model(
            [encoder_input, condition_input], [z_mean, z_log_var, z],
            name="cvae_encoder",
        )

        decoder_input    = tf.keras.layers.Input(shape=(latent_dim,))
        condition_input2 = tf.keras.layers.Input(shape=(condition_dim,))
        x = tf.keras.layers.Concatenate(axis=-1)([decoder_input, condition_input2])
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        output = tf.keras.layers.Dense(input_dim, activation="sigmoid")(x)
        self.decoder = tf.keras.Model(
            [decoder_input, condition_input2], output,
            name="cvae_decoder",
        )

    def train_step(self, data):
        X_batch, y_batch = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([X_batch, y_batch])
            reconstruction       = self.decoder([z, y_batch])
            reconstruction_loss  = tf.reduce_mean(
                keras.losses.mse(X_batch, reconstruction)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_latent_space(vae_model, X_data, Y_labels, title="Latent Space"):
    z_mean, _, _ = vae_model.encoder.predict(X_data, verbose=0)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=Y_labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("latent_space.png", dpi=150)
    plt.show()


def plot_reconstruction(vae_model, X_test, sample_index, title=""):
    z_mean, _, z = vae_model.encoder.predict(X_test, verbose=0)
    reconstruction = vae_model.decoder.predict(z, verbose=0)

    plt.figure(figsize=(12, 4))
    plt.plot(WAVENUMBER_AXIS, X_test[sample_index],
             label="Original", alpha=0.75)
    plt.plot(WAVENUMBER_AXIS, reconstruction[sample_index],
             label="Reconstructed", alpha=0.75)
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title or f"Reconstruction — sample {sample_index}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reconstruction_{sample_index}.png", dpi=150)
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["reconstruction_loss"], label="Reconstruction Loss")
    plt.plot(history.history["kl_loss"],             label="KL Divergence")
    plt.plot(history.history["loss"],                label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("VAE Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vae_training_history.png", dpi=150)
    plt.show()


# ── Downstream latent classifier ──────────────────────────────────────────────

def train_latent_classifier(encoder, X_test, Y_test, n_splits=N_FOLDS_DOWNSTREAM):
    """Gradient-boosted tree classifier on VAE latent representations."""
    encoded = encoder.predict(X_test, verbose=0)
    Z = np.asarray(encoded[0])     # z_mean

    gb_classifier = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)

    true_labels, pred_labels, pred_probs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(Z, Y_test), 1):
        Z_train, Z_test = Z[train_idx], Z[test_idx]
        Y_train, Y_fold_test = Y_test[train_idx], Y_test[test_idx]

        gb_classifier.fit(Z_train, Y_train)
        preds = gb_classifier.predict(Z_test)
        probs = gb_classifier.predict_proba(Z_test)

        true_labels.extend(Y_fold_test)
        pred_labels.extend(preds)
        pred_probs.extend(probs)

    print("\nLatent Classifier Report:")
    print(classification_report(true_labels, pred_labels,
                                 target_names=CLASS_NAMES))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Latent Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig("latent_classifier_confusion.png", dpi=150)
    plt.show()

    return true_labels, pred_labels


# ── Dimensionality reduction visualisation ────────────────────────────────────

def plot_tsne(X_data, Y_labels, perplexity=110):
    tsne    = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    X_tsne  = tsne.fit_transform(X_data)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=Y_labels,
                     title="t-SNE Visualisation",
                     labels={"x": "tSNE_1", "y": "tSNE_2"})
    fig.update_traces(marker=dict(size=8))
    fig.show()


def plot_umap(X_data, Y_labels, n_neighbors=170):
    reducer = umap.UMAP(n_components=2, init="random",
                        random_state=RANDOM_STATE, n_neighbors=n_neighbors)
    X_umap  = reducer.fit_transform(X_data)
    fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=Y_labels,
                     title="UMAP Visualisation",
                     labels={"x": "UMAP_1", "y": "UMAP_2"})
    fig.update_traces(marker=dict(size=8))
    fig.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loading import load_raw_data, preprocess_spectra

    raw_spectra, labels_df = load_raw_data()
    X_processed = preprocess_spectra(raw_spectra)   # shape (N, 901)
    Y_labels    = labels_df["Key"].to_numpy()

    # Train/test split (augmented train set recommended — see data_loading.py)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_processed, Y_labels, test_size=0.2, random_state=RANDOM_STATE,
        stratify=Y_labels,
    )

    # t-SNE / UMAP visualisation of raw spectra
    plot_tsne(X_processed, Y_labels)
    plot_umap(X_processed, Y_labels)

    # β-VAE (set beta=0 for standard VAE)
    encoder = build_dense_encoder(X_processed.shape[1], LATENT_DIM)
    decoder = build_dense_decoder(LATENT_DIM, X_processed.shape[1])

    beta_vae = BetaVAE(encoder, decoder, beta=BETA)
    beta_vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )
    beta_vae.history = beta_vae.fit(
        X_train, epochs=N_EPOCHS_BETA_VAE, batch_size=BATCH_SIZE, verbose=2
    )

    plot_training_history(beta_vae.history)
    plot_latent_space(beta_vae, X_test, Y_test, title="β-VAE Latent Space")
    plot_reconstruction(beta_vae, X_test, sample_index=1,
                        title="β-VAE Reconstruction — Class 1")

    train_latent_classifier(beta_vae.encoder, X_test, Y_test)