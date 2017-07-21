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
from scipy.stats import multivariate_normal


def build_generator():
    z = Input(shape=(latent_dim,), name='z')
    h = Dense(units=5, activation='relu')(z)
    h = Dense(units=7, activation='relu')(h)
    h = Dense(units=10, activation='relu')(h)
    h = Dense(units=10, activation='relu')(h)
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
    x_real = next_batch(num_samples)

    plt.scatter(x_real[:,0], x_real[:,1], label='x_real')
    plt.scatter(x_gen[:,0], x_gen[:,1], label='x_gen')
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
    #densities = distribution.pdf(x_gen)

    densities = multivariate_normal.pdf(x_gen, mean=data_mean, cov=np.eye(data_dim)*data_std) + multivariate_normal.pdf(x_gen, mean=data_mean2, cov=np.eye(data_dim)*data_std2)

    sort_ind = np.argsort(densities)
    #densities_max = np.max(densities, axis=1)
    #grad_max = np.max(grad, axis=1)

    #indices_sorted = np.argsort(grad_max)
    #plt.scatter(y=grad_max, x=densities_max)
    #plt.ylabel("max(gradient)")
    #plt.xlabel("max(density)")
    #plt.show()
    #plt.close()

    #densities_norm = np.linalg.norm(densities, ord=2, axis=1)
    grad_norm = np.linalg.norm(grad, ord=2, axis=1)
    #plt.scatter(y=grad_norm, x=densities_norm)
    #plt.ylabel("l2_norm(gradient)")
    #plt.xlabel("l2_norm(density)")
    #plt.show()
    #plt.close()
    plt.plot(densities[sort_ind], grad_norm[sort_ind])
    plt.xlabel("density")
    plt.ylabel("gradient")
    plt.show()


def next_batch(batch_size=100):
    X1 = np.random.normal(data_mean, data_std, (int(batch_size / 2), 2))
    X2 = np.random.normal(data_mean2, data_std2, (int(batch_size / 2), 2))
    return np.concatenate((X1,X2))


data_mean = [0, 10]
data_std = [5, 5]
data_mean2 = [-5, 5]
data_std2 = [5, 5]

batch_size = 128
data_dim = 2
latent_dim = 100

plot_points(1000)
# Create a line of points connecting the centers of the 2 modes
#x0 = data_mean[0]
#y0 = data_mean[1]
#
#x1 = data_mean2[0]
#y1 = data_mean2[1]
#
#x_range = np.linspace(x0, x1, num=n)
#l = (y1 - y0) / (x1 - x0)
#y_range = y0 + l*(x_range-x0)
#x_test = np.asarray([i for i in zip(x_range, y_range)])

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
num_iter = 10**4

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

#plot_points(1000)
plot_grad_vs_density(10000)
