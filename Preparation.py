from glob import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import set_image_dim_ordering
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import adam

def normalize(x):
    x /= 255
    return x
def init (self):
	return self

def load_dataset(type):
    if type is 'train' or type is 'validation' or type is 'test':
        path = 'data/'
        path += type
        path += '/*.jpg'
        set_image_dim_ordering('th')
        x = []
        y = []
        for img_path in glob(path):
            img = load_img(img_path, target_size= emotion_target_size, grayscale=True)
            y_post = img_path.find('_') + 1
            x.append(img_to_array(img))
            y.append(img_path[y_post])
        x = np.array(x)
        y = np.array(y)
        x = normalize(x)
        y = to_categorical(y, 7)
        return x, y
    else:
        print('type is invalid, please use "train", "validation", or "test"')

def train(x, y, x_val=None, y_val=None, save_model=False, lr=1e-3, epoch=50, rotation_range=0.0, width_shift_range=0.0,
          height_shift_range=0.0,
          horizontal_flip=True, vertical_flip=False):

    model = Sequential()
    self.model.add(Conv2D(16, (3, 3), padding='same', input_shape=x.shape[1:]))
	self.model.add(activation ('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.25))
    self.model.add(Conv2D(32, (3, 3), padding='same'))
	self.model.add(activation ('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.25))
    self.model.add(Conv2D(64, (3, 3), padding='same'))
	self.model.add(activation ('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.25))
    self.model.add(Conv2D(128, (3, 3), padding='same'))
	self.model.add(activation ('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.25))
    self.model.add(Flatten())

    self.model.add(Dense(1280))
	self.model.add(activation ('relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.5))
    self.model.add(Dense(640))
	self.model.add(activation ('relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.5))

    self.model.add(Dense(7, activation='softmax'))

    self.model.summary()
    self.model.compile(loss='categorical_crossentropy', optimizer=adam(lr=lr), metrics=['accuracy'])

    dataGenerator = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range, horizontal_flip=horizontal_flip,
                                       vertical_flip=vertical_flip)
    dataGenerator.fit(x)

    self.model.fit_generator(dataGenerator.flow(x, y, batch_size=128), validation_data=(x_val, y_val),
                        epochs=epoch, workers=10, verbose=2)

    if save_model:
        self.model.save('model_final_fix.h5')
        self.model.save_weights('weight_model_final_fix.h5')

    scores = model.evaluate(x, y, verbose=0)
    print('--- Result---')
    scores = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss: %.4f' % scores[0])
    print('Validation accuracy: %.3f%%' % (scores[1] * 100))
