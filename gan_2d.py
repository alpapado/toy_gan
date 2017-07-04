import keras
from keras.layers import Input, Dense, Reshape, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.random as rand
from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import norm


def build_generator():
    z = Input(shape=(latent_dim,), name='z')
    h = Dense(units=10, activation='relu')(z)
    x_gen = Dense(units=data_dim, activation='linear', name='x_gen')(h)
    generator = Model(inputs=z, outputs=x_gen)
    return generator


def build_discriminator():
    x = Input(shape=(data_dim,))
    h = Dense(units=10, activation='relu')(x)
    fake_prob = Dense(units=1, activation='sigmoid')(h)
    discriminator = Model(inputs=x, outputs=fake_prob)
    return discriminator


def plot_points(num_samples=1000):
    z = np.random.uniform(-1, 1, size=(num_samples, latent_dim))
    x_gen = generator.predict(z)
    x_real = rand.normal(size=(num_samples, data_dim), loc=data_mean, scale=data_std)

    plt.scatter(x_gen[:,0], x_gen[:,1], label='x_gen')
    plt.scatter(x_real[:,0], x_gen[:,1], label='x_real')
    plt.legend(loc='upper left')
    plt.show()


def plot_grad_vs_density(num_samples=100):
    gen_out = generator.get_layer(name='x_gen').output
    gen_in = generator.get_layer(name='z').input
    compute_gradient = K.gradients(gen_out, gen_in)
    sess = K.get_session()

    distribution = norm(loc=data_mean, scale=data_std)
    z = np.random.uniform(-1, 1, size=(num_samples, latent_dim))
    x_gen = generator.predict(z)

    grad = sess.run(compute_gradient, feed_dict={gen_in:z})[0]
    densities = distribution.pdf(x_gen)

    densities_max = np.max(densities, axis=1)
    grad_max = np.max(grad, axis=1)

    #indices_sorted = np.argsort(grad_max)
    #plt.scatter(y=grad_max, x=densities_max)
    #plt.ylabel("max(gradient)")
    #plt.xlabel("max(density)")
    #plt.show()
    #plt.close()

    densities_norm = np.linalg.norm(densities, ord=2, axis=1)
    grad_norm = np.linalg.norm(grad, ord=2, axis=1)
    plt.scatter(y=grad_norm, x=densities_norm)
    plt.ylabel("l2_norm(gradient)")
    plt.xlabel("l2_norm(density)")
    plt.show()
    plt.close()


if __name__ == "__main__":
    data_mean = (0, 100)
    data_std = (1, 5)
    batch_size = 128
    data_dim = 2
    latent_dim = 100

    # Build discriminator
    discriminator = build_discriminator()

    # Build generator
    generator = build_generator()

    # Build combined generator-discriminator
    discriminator.trainable = False
    z = Input(shape=(latent_dim,))
    x_gen = generator(z)
    fake_prob = discriminator(x_gen)
    combined = Model(inputs=z, outputs=fake_prob)

    # Compile all models
    opt = Adam(lr=0.0002, beta_1=0.5)
    generator.compile(optimizer=opt, loss='binary_crossentropy')
    combined.compile(optimizer=opt, loss='binary_crossentropy')

    discriminator.trainable = True
    discriminator.compile(optimizer=opt, loss='binary_crossentropy')

    # Let's train
    num_iter = 200

    y_fake = np.zeros(shape=(batch_size, 1))
    y_real = np.ones(shape=(batch_size, 1))

    for i in range(num_iter):
        x_real = rand.normal(size=(batch_size, data_dim), loc=data_mean, scale=data_std)

        z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
        x_fake = generator.predict(z)

        x = np.concatenate((x_real, x_fake))
        y = np.concatenate((y_real, y_fake))

        # Train discriminator
        d_loss = discriminator.train_on_batch(x, y)

        # Train generator
        z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(x=z, y=y_real)
        discriminator.trainable = True

        if (i+1) % 100 == 0:
            print("Iteration %d - g_loss = %f - d_loss = %f" % (i+1, g_loss, d_loss))

    plot_points(10000)
    plot_grad_vs_density(10000)
