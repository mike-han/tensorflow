from tabnanny import verbose
from wsgiref import validate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

keras = tf.keras
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/tmp/horse-or-human/', target_size=(300, 300), batch_size=32, class_mode='binary')

validation_generator = validation_datagen.flow_from_directory('/tmp/validation-horse-or-human/', target_size=(300, 300), batch_size=32, class_mode='binary')


model = keras.Sequential()

model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['acc'])

history = model.fit(train_generator, epochs=15, verbose=1, validation_data=validation_generator, validation_steps=8)

model.save('./horse-or-human.h5')