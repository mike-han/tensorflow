from operator import mod
import numpy
import tensorflow as tf
keras = tf.keras
mnist = tf.keras.datasets.mnist

class TensorCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss') < 0.4):
      print("\nLoss is low so cancelling training!")
      self.model.stop_training = True

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer = tf.optimizers.Adam(),loss = tf.losses.sparse_categorical_crossentropy,metrics = ['accuracy'])
model.fit(train_images/255, train_labels, epochs=5, callbacks=[TensorCallback()])

model.evaluate(test_images/255, test_labels)
print(numpy.argmax(model.predict(test_images[0:1]/255), axis=1))
print(test_labels[0])