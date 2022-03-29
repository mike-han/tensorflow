import tensorflow as tf

model = tf.keras.models.load_model('./horse-or-human.h5')
model.summary()