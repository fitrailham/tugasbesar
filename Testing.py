from keras.models import load_model
from glob import glob
import numpy as np
from keras.backend import set_image_dim_ordering
from keras.preprocessing.image import load_img, img_to_array

prediction = []
path = 'data/test/*.jpg'
model = load_model('model_final_fix.h5')

for img_path in glob(path):
    x = []
    set_image_dim_ordering('th')
    img = load_img(img_path, grayscale=True)
    x.append(img_to_array(img))
    x = np.array(x)
    x /= 255
    predict = model.predict(x)
    prediction.append(np.argmax(predict[0]))

print('Prediction')
for isi in prediction:
    print(isi)