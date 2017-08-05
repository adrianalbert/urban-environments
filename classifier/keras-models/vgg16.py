from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.engine import Layer
from keras import backend as K
from keras.optimizers import SGD

import cv2, numpy as np

def vgg16(n_classes=1000, input_shape=(224,224,3), fcn=False, add_top=True):
    model = Sequential()

    if fcn:
        model.add(ZeroPadding2D((1,1),input_shape=(None,None,input_shape[2])))
    else:
        model.add(ZeroPadding2D((1,1),input_shape=input_shape))

    model.add(Convolution2D(64, 3, 3, activation='relu', name="conv1_1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="conv1_2"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="conv2_1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="conv2_2"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="conv3_1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="conv3_2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="conv3_3"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv4_1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv4_2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv4_3"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv5_1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv5_2"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="conv5_3"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if add_top:
        if fcn:
            model.add(Convolution2D(4096,7,7,activation="relu",name="dense_1"))
            model.add(Convolution2D(4096,1,1,activation="relu",name="dense_2"))
            model.add(Convolution2D(n_classes,1,1,name="dense8"))
            model.add(Softmax4D(axis=1,name="softmax"))
        else:
            model.add(Flatten())
            model.add(Dense(4096, activation='relu', name="dense6"))
            model.add(Dropout(0.5))
            model.add(Dense(4096, activation='relu', name="dense7"))
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax', name="dense8"))

    return model


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s


# if __name__ == "__main__":
#     im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#     im[:,:,0] -= 103.939
#     im[:,:,1] -= 116.779
#     im[:,:,2] -= 123.68
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)

#     # Test pretrained model
#     model = vgg16('vgg16_weights.h5')
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='categorical_crossentropy')
#     out = model.predict(im)
#     print np.argmax(out)