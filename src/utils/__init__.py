import numpy as np
from keras.preprocessing import image

def get_image_for_predict(path, target_size = (150, 150)):
  img = image.load_img(path, target_size)
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis = 0)
  return img

