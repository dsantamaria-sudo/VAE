#%%
#IMPORT ALL LIBRARIES

from   PIL                       import Image
from   os                        import listdir
from   numpy                     import asarray
from   numpy                     import vstack
from   keras.preprocessing.image import img_to_array
from   keras.preprocessing.image import load_img
from   numpy                     import savez_compressed
import tensorflow                as tf; tf.compat.v1.disable_eager_execution()
from   keras                     import backend as K
from   keras.layers              import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape, Dropout
from   keras.models              import Model
from   keras.losses              import binary_crossentropy
from   tensorflow                import keras
from   tensorflow.keras          import layers
import tensorflow                as tf
import numpy                     as np
import matplotlib.pyplot         as plt
import random
import os
import cv2
import math
np.random.seed(25)

#SET UP GPU COMPUTING
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.executing_eagerly()

#SET IMG DIRECTORIES
directory_open = r'C:\Users\Usuario\Desktop\Fotos'

#LOAD AND SHUFFLE IMAGES

list_dir = listdir(directory_open)
random.shuffle(list_dir)

#RESIZE AND TRANSFORM IMAGES TO ARRAYS

img_sample = 800 # size of the sample 
img_array=[]	

for filename in list_dir[0:img_sample]:

    pixels = load_img(f"{directory_open}\{filename}")
    pixels = pixels.resize((512, 512))
    pixels = img_to_array(pixels)/255

    img_array.append(pixels)

img_array  = np.array(img_array)


#SET TEST AND TRAIN SAMPLES

train_size  = 600
X_train_new = img_array[0:train_size]
X_test_new  = img_array[train_size:]


#SET DIMENSIONS OF THE INPUT AND LATENT SPACE

img_height   = X_train_new.shape[1]   
img_width    = X_train_new.shape[2]
num_channels = X_train_new.shape[3] 
input_shape  = (img_height, img_width, num_channels)
latent_dim   = 2   

#ENCODER NEURAL NETWORK

encoder_input = Input(shape=input_shape)
encoder_conv  = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(encoder_input)
encoder_conv  = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(encoder_conv)
encoder       = Flatten()(encoder_conv)

mu            = Dense(latent_dim)(encoder)
sigma         = Dense(latent_dim)(encoder)

#LATENT SPACE

def compute_latent(x):
    mu, sigma = x
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps

latent_space  = Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])

conv_shape    = K.int_shape(encoder_conv)

#DECODER NEURAL NETWORK

decoder_input = Input(shape=(latent_dim,))
decoder       = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
decoder       = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
decoder_conv  = Conv2DTranspose(filters=16, kernel_size=3, padding='same', activation='relu')(decoder)
decoder_conv  = Conv2DTranspose(filters=8, kernel_size=3, padding='same', activation='relu')(decoder_conv)
decoder_conv  = Conv2DTranspose(filters=num_channels, kernel_size=3, padding='same', activation='sigmoid')(decoder_conv)

#VAE MODEL BUILDING

encoder = Model(encoder_input, latent_space)
decoder = Model(decoder_input, decoder_conv)
vae     = Model(encoder_input, decoder(encoder(encoder_input)))

#LOSS FUNCTION DEFINITION

def kl_reconstruction_loss(true, pred):

    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    kl_loss             = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss             = K.sum(kl_loss, axis=-1)
    kl_loss            *= -0.5
    return K.log(K.mean(reconstruction_loss + kl_loss))

#MODEL COMPILATION

vae.compile(optimizer= 'adam', loss=kl_reconstruction_loss)

#-----------------------------------------TRAINING--------------------------------------------------------------------

#/!\ TO TEST THE MODEL WITH SAVED WEIGHTS UNNCOMENT THE FOLLOWING LINES AND LOAD SAVED WEIGHTS:

#vae.load_weights('vae.h5')

history = vae.fit(x=X_train_new, y=X_train_new, epochs=500, batch_size=24, validation_data=(X_test_new,X_test_new))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#VAE PREDICTIONS

encoded    = encoder.predict(X_test_new)
new_images = decoder.predict(encoded)


# %%
#CHECK VAE PREDICTION
# i = any number between 0 and test size

tf.keras.preprocessing.image.array_to_img(new_images[i]*255)

# %%
