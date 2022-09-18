import numpy as np
from keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Concatenate
from keras import Model
from matplotlib import pyplot as plt


# define the standalone discriminator model
def define_discriminator(img_shape=(28, 28, 1), n_classes=10, z_dim=100):
    img = Input(shape=img_shape, name='input image')  # input image 28x28x1

    label = Input(shape=(1,), name='input label')  # input label [0, 9]
    label_embedding = Embedding(n_classes, z_dim)(label)
    label_dense = Dense(img_shape[0] * img_shape[1])(label_embedding)
    label_reshape = Reshape((img_shape[0], img_shape[1], 1))(label_dense)

    x = Concatenate()([img, label_reshape])
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[img, label], outputs=output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


# define the standalone generator model
def define_generator(z_dim=100, n_classes=10):
    z = Input(shape=(z_dim,), name='input noise')  # input noise vector
    z_dense = Dense(7 * 7 * 128)(z)
    z_reshape = Reshape((7, 7, 128))(z_dense)

    label = Input(shape=(1,), name='input label')  # input label
    label_embedding = Embedding(n_classes, z_dim)(label)
    label_dense = Dense(7 * 7 * 1)(label_embedding)
    label_reshape = Reshape((7, 7, 1))(label_dense)

    x = Concatenate()([z_reshape, label_reshape])
    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    output = Conv2D(1, 8, activation='sigmoid', padding='same')(x)

    model = Model(inputs=[z, label], outputs=output)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False

    # gen inputs (noise, label) --> gen output (28x28x1 img)
    z, label = g_model.input
    img = g_model.output

    # disc inputs (img, label) --> disc output (real or fake)
    gan_output = d_model([img, label])

    # compile model
    model = Model(inputs=[z, label], outputs=gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load and prepare mnist training images
def load_real_samples():
    # load mnist dataset
    (x, labels), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = np.expand_dims(x, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0
    # reshaping labels
    labels = np.expand_dims(labels, axis=-1)
    return X, labels


# select real samples
def generate_real_samples(dataset, n_samples):
    x, labels = dataset
    # choose random instances
    i = np.random.randint(0, x.shape[0], n_samples)
    # retrieve selected images
    X = x[i]
    # retrieve selected labels
    labels = labels[i]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, labels, y


# generate points in latent space as input for the generator
def generate_latent_points(z_dim, n_samples, n_classes=10):
    # generate points in the latent space
    z = np.random.randn(n_samples, z_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, size=(n_samples, 1))
    return z, labels


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, z_dim, n_samples, balanced=False):
    # generate points in latent space
    z, labels = generate_latent_points(z_dim, n_samples)
    if balanced:
        labels = np.vstack([np.arange(10)] * 10).T.reshape((-1, 1))  # 10 labels of each (10 * 10 = 100)
    # predict outputs
    X = g_model.predict([z, labels])
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, labels, y


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, z_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # create inverted labels for the fake samples
    y_gan = np.ones((n_batch, 1))
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, labels_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, labels_fake, y_fake = generate_fake_samples(g_model, z_dim, half_batch)
            # create training set for the discriminator
            X, labels, y = np.vstack((X_real, X_fake)), np.vstack((labels_real, labels_fake)), np.vstack(
                (y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch([X, labels], y)
            # prepare points in latent space (and labels) as input for the generator
            X_gan, labels_gan = generate_latent_points(z_dim, n_batch)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([X_gan, labels_gan], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, z_dim)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    #     plt.show()
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, z_dim, n_samples=100):
    # prepare real samples
    X_real, labels_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
    # prepare fake examples
    x_fake, labels_fake, y_fake = generate_fake_samples(g_model, z_dim, n_samples, balanced=True)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate([x_fake, labels_fake], y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    print(labels_fake.reshape((10, 10)))
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


# size of the latent space
z_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(z_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, z_dim, n_epochs=100)
