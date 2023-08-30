#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # The Sequential model API

#  ## Coding tutorials
#  #### [1. Building a Sequential model](#coding_tutorial_1)
#  #### [2. Convolutional and pooling layers](#coding_tutorial_2)
#  #### [3. The compile method](#coding_tutorial_3)
#  #### [4. The fit method](#coding_tutorial_4)
#  #### [5. The evaluate and predict methods](#coding_tutorial_5)

# ***
# <a id="coding_tutorial_1"></a>
# ## Building a Sequential model

# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax


# #### Build a feedforward neural network model

# In[ ]:


# Build the Sequential feedforward neural network model


# In[ ]:


# Print the model summary


# ***
# <a id="coding_tutorial_2"></a>
# ## Convolutional and pooling layers

# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# #### Build a convolutional neural network model

# In[5]:


# Build the Sequential convolutional neural network model

model = Sequential([
    Conv2D(16, (3,3), activation = 'relu', input_shape=(28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10, activation = 'softmax', name = 'SoftMax')
])


# In[6]:


# Print the model summary

model.summary()


# ***
# <a id="coding_tutorial_3"></a>
# ## The compile method

# #### Compile the model

# In[11]:


# Define the model optimizer, loss function and metrics

opt = tf.keras.optimizers.Adam(learning_rate = 0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(
    optimizer = opt,
    loss = 'sparse_categorical_crossentropy',
    metrics = [acc, mae]
)


# In[ ]:


# Print the resulting model attributes


# ***
# <a id="coding_tutorial_4"></a>
# ## The fit method

# In[12]:


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# #### Load the data

# In[13]:


# Load the Fashion-MNIST dataset

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()


# In[14]:


# Print the shape of the training data

print(train_images.shape)


# In[15]:


# Define the labels

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(train_labels[0])


# In[18]:


# Rescale the image values so that they lie in between 0 and 1.

train_images = train_images/255.
test_images = test_images/255.


# In[19]:


# Display one of the images

i = 0 
img = train_images[i,:,:]
plt.imshow(img)
plt.show()
print(f"labels: {labels[train_labels[i]]}")


# #### Fit the model

# In[23]:


# Fit the model

history = model.fit(train_images[...,np.newaxis], train_labels, epochs = 8, batch_size=256)


# #### Plot training history

# In[26]:


# Load the history into a pandas Dataframe

df = pd.DataFrame(history.history)
df.head()


# In[27]:


# Make a plot for the loss

loss_plot = df.plot(y = 'loss', title = 'Loss vs. Epochs', legend = False)
loss_plot.set(xlabel = 'Epochs', ylabel = 'Loss')


# In[ ]:


# Make a plot for the accuracy


# In[ ]:


# Make a plot for the additional metric


# ***
# <a id="coding_tutorial_5"></a>
# ## The evaluate and predict methods

# In[28]:


import matplotlib.pyplot as plt
import numpy as np


# #### Evaluate the model on the test set

# In[32]:


# Evaluate the model

test_loss, test_accuracy, test_mae = model.evaluate(test_images[...,np.newaxis], test_labels, verbose = 2)


# #### Make predictions from the model

# In[34]:


# Choose a random test image

random_inx = np.random.choice(test_images.shape[0])

inx = 30

test_image = test_images[inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[random_inx]]}")


# In[41]:


# Get the model predictions

predictions = model.predict(test_image[np.newaxis,...,np.newaxis])
print(f"Model prediction:{labels[np.argmax(predictions)]}")

