
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
chanDim = -1
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size =  (3,3), input_shape = (64,64,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))

model.add(Conv2D(filters = 16, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 16, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2, 2)))

model.add(Conv2D(filters = 32, kernel_size =  (3,3), padding = 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 32, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 32, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))


model.add(Conv2D(filters = 128, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()
import os
fname = "car_weights-{val_acc:.4f}-{val_loss:.4f}.hdf5"
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(fname, monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255,
				shear_range = 0.3,	
				zoom_range = 0.3,
    				rotation_range=45,
    				width_shift_range=0.3,
   				height_shift_range=0.3,
          vertical_flip = True,                         
				fill_mode = 'nearest')
train_generator = train_datagen.flow_from_directory(
        '/content/train',
        target_size=(64, 64),
        batch_size=64,
	shuffle=True,
        color_mode = 'rgb',
        class_mode='categorical',
        save_format='jpeg')
dev_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = dev_datagen.flow_from_directory(
                '/content/dev',
        target_size=(64, 64),
        batch_size=64,
        color_mode ='rgb',
	shuffle=True,
        class_mode='categorical')

from keras.optimizers import SGD,Adam

optimizer =  SGD(0.0005, decay= 0.01/40, momentum = 0.9, nesterov = 'true')
model.compile( optimizer = optimizer , loss='categorical_crossentropy', metrics=['accuracy'])
epoch = 100


H = model.fit_generator(
        train_generator,
	callbacks = callbacks,
        steps_per_epoch=65,
        epochs=epoch,
        verbose = 1,
        validation_data=validation_generator,
        validation_steps=8)

import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use("ggplot")
fig = plt.figure()


plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epoch), H.history["val_acc"], label="val_acc")
title = "0.01"
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")

plt.ylabel("Loss/Accuracy")
plt.legend()
fig.savefig('image.png', dpi=fig.dpi)
plt.show()