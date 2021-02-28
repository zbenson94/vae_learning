

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

class vae(tf.keras.Model):

    """Variational autoencoder."""

    def __init__(self, latent_dim, nNodes, input_shape):
        super(vae, self).__init__()

        self.latent_dim = latent_dim
        self.encoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=nNodes,activation=tf.nn.relu),
                tf.keras.layers.Dense(units=nNodes,activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=nNodes,activation=tf.nn.relu),
                tf.keras.layers.Dense(units=nNodes,activation=tf.nn.relu),
                tf.keras.layers.Dense(units=np.prod(input_shape)),
                tf.keras.layers.Reshape(target_shape=input_shape),
            ]
        )



    @tf.function
    def sample(self, eps=None,apply_sigmoid=False):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

class vae_CNN2D(tf.keras.Model):

    """Convolutional Variational autoencoder."""

    def __init__(self, latent_dim, nFilters, input_shape):
        super(vae_CNN2D, self).__init__()

        self.latent_dim = latent_dim


        self.encoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=nFilters, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=2*nFilters, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(input_shape), activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=input_shape),
                tf.keras.layers.Conv2DTranspose(
                    filters=2*nFilters, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=nFilters, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )



    @tf.function
    def sample(self, eps=None,apply_sigmoid=False):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class vae_CNN3D(tf.keras.Model):

    """Convolutional Variational autoencoder."""

    def __init__(self, latent_dim, nFilters, input_shape):
        super(vae_CNN2D, self).__init__()

        self.latent_dim = latent_dim


        self.encoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv3D(
                    filters=nFilters, kernel_size=3, strides=(2, 2, 2), activation='relu'),
                tf.keras.layers.Conv3D(
                    filters=2*nFilters, kernel_size=3, strides=(2, 2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder    = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(input_shape), activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=input_shape),
                tf.keras.layers.Conv3DTranspose(
                    filters=2*nFilters, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv3DTranspose(
                    filters=nFilters, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv3DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )



    @tf.function
    def sample(self, eps=None,apply_sigmoid=False):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=apply_sigmoid)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



def compute_loss(model, x, sigma=0.1, beta=1.0,apply_sigmoid=True):
    # Gets the mean and logvar used for kl divergence and latent space
    mean, logvar = model.encode(x)
    var          = tf.keras.activations.softplus(logvar)
    # Need to calculate reconstruction loss by creating PROBABILITY DISTRIBUTION FUNCTIONS
    
    z       = model.reparameterize(mean, logvar)
    
    probs   = model.decode(mean,apply_sigmoid=apply_sigmoid)
    
    pz      = tfd.Independent(tfd.Normal(loc=probs,scale=sigma*tf.ones(probs.shape)),3)

    logpz     = tfd.MultivariateNormalDiag(tf.zeros(mean.shape),tf.ones(logvar.shape))
    logqz_x   = tfd.MultivariateNormalDiag(mean,var)


    likelihood    = tf.reduce_mean(pz.log_prob(x))
    kl_divergence = tfd.kl_divergence(logqz_x,logpz)
    
    return -tf.reduce_mean(likelihood  - beta*kl_divergence)


@tf.function
def train_step(model, x, optimizer):

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

