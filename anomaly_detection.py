#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
def convert_video_to_images(img_folder, filename='assignment2_video.avi'):
    """
    Converts the video file (assignment2_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')
    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # Instantiate the video object.
    video = cv2.VideoCapture(filename)
    # Check if the video is opened successfully
    

    if not video.isOpened():
        print("Error opening video file")

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    if i:
        print(f'Video converted\n{i} images written to {img_folder}')


# In[2]:


from PIL import Image
from glob import glob
import numpy as np
import os
def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.
    The images are returned both as a Numpy array of flattened images (i.e. the images with the 3-d shape (im_
    Arguments
    ---------
    img_dir : (string) the directory where the images are stored.
    im_width : (int) The desired width of the image.
    The default value works well.
    im_height : (int) The desired height of the image.
    The default value works well.
    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized images.
    """
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    
    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()
        
    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    
    return X, images


# In[3]:


convert_video_to_images('anomalydetection')
X, images = load_images('anomalydetection')


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.4, random_state = 142)


# In[14]:


X_train.shape


# In[15]:


from tensorflow import keras
from keras import layers


# In[16]:


encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(7920,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(7920, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)


# In[17]:


encoder = keras.Model(input_img, encoded)


# In[18]:


# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


# In[19]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[20]:


autoencoder.summary()


# In[21]:


x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
x_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
x_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)


# In[22]:


autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))


# In[26]:


import tensorflow as tf
def threshold(autoencoder, X_train):
  reconstructions = autoencoder.predict(X_train)
  reconstruction_loss = tf.keras.losses.msle(reconstructions, X_train)
  threshold = np.mean(reconstruction_loss.numpy()) + np.std(reconstruction_loss.numpy())
  return threshold


# In[27]:


import pandas as pd
def predict(autoencoder, X_test, threshold):
  predictions = autoencoder.predict(X_test)
  errors = tf.keras.losses.msle(predictions, X_test)
  anomaly_mask = pd.Series(errors) > threshold
  preds = anomaly_mask.map(lambda x: False if x == True else True)
  return preds


# In[28]:


threshold = threshold(autoencoder, X_train)


# In[29]:


predictions = predict(autoencoder, X_test, threshold)


# In[30]:


print(predictions)


# In[31]:


print(predictions.value_counts())


# In[ ]:




