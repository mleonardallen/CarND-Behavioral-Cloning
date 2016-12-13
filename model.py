import numpy as np
import pandas as pd
import json
import matplotlib.image as img
from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

# 1. Read Data from Driving Log
driving_log = pd.read_csv('driving_log.csv', index_col=False)
driving_log.columns = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']

# 2. Prepare Data for Generator
X_train, y_train = [], []
for index, row in driving_log.iterrows():
    X_train.append(row['Center Image'])
    y_train.append(row['Steering Angle'])
      
    # left and right steering angles
    C = row['Steering Angle']
    L = C + 0.07
    R = C - 0.07
        
    X_train.append(row['Left Image'].strip())
    y_train.append(L)
        
    X_train.append(row['Right Image'].strip())
    y_train.append(R)
  
X_train, y_train = np.array(X_train), np.array(y_train)

# 3. Generator for Image/Steering Angle tuples
class SteeringDataGenerator(ImageDataGenerator):

    def flow(self, X, y=None, batch_size=32, shuffle=False, seed=None, flip_prob=0):
        return SteeringIterator(X, y, batch_size=batch_size, shuffle=shuffle, seed=seed, flip_prob=flip_prob)

class SteeringIterator(Iterator):

    def __init__(self, X, y, batch_size=32, shuffle=False, seed=None, flip_prob=0):

        self.X = X
        self.y = y
        self.flip_prob = flip_prob

        super(SteeringIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x, batch_y = None, None
        for batch_idx, source_idx in enumerate(index_array):
            # load the image from path
            x = img.imread(self.X[source_idx])
            y = self.y[source_idx]

            # now that we have our image shape, initialize the batch
            if batch_x is None:
                batch_x = np.zeros(tuple([current_batch_size] + list(x.shape)))
                batch_y = np.zeros(current_batch_size)

            # random horizontal flip
            if np.random.choice([True, False], p=[self.flip_prob, 1.-self.flip_prob]):
                x = flip_axis(x, 1)
                y *= -1

            # store image in batch
            batch_x[batch_idx] = x
            batch_y[batch_idx] = y
            
        return batch_x, batch_y

# 4. Define Pipeline
input_shape = (160, 320, 3)
cropping = ((60, 0), (0, 0))

def resize(X):
    import tensorflow
    return tensorflow.image.resize_images(X, (50, 120))

model = Sequential([
    # Preprocess
    Cropping2D(cropping=cropping, input_shape=input_shape),
    Lambda(resize),
    BatchNormalization(axis=1),
    # Network
    Convolution2D(24, 5, 5, border_mode='same', activation='relu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    Convolution2D(36, 5, 5, border_mode='same', activation='relu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    Convolution2D(48, 3, 3, border_mode='same', activation='relu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
    MaxPooling2D(border_mode='same'),
    SpatialDropout2D(0.2),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# 5. Train Model
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
nb_val_samples = len(y_val)
datagen = SteeringDataGenerator()
model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=128, shuffle=True, flip_prob=0.5),
    samples_per_epoch=len(y_train),
    validation_data=datagen.flow(X_val, y_val, batch_size=128, shuffle=True),
    nb_val_samples=nb_val_samples,
    nb_epoch=5
)

# 6. Save Model/Weights
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights('model.h5')
