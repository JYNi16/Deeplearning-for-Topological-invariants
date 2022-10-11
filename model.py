import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, losses

class CNN_TI(Model):

    def __init__(self):
        super(CNN_TI, self).__init__()
        self.c1 = layers.Conv2D(40, kernel_size=(2, 2), activation='relu', input_shape=(33, 2, 1))
        self.c2 = layers.Conv2D(1, kernel_size=(1, 1), activation='relu')
        self.f1 = layers.Flatten()
        self.dr1 = layers.Dropout(0.005)
        self.d1 = layers.Dense(16, activation="relu")
        self.d2 = layers.Dense(3, activation="linear")

    def call(self,x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.f1(x)
        x = self.dr1(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

if __name__=="__main__":
    model = CNN_TI()
    a = tf.ones((1,33, 2, 1))
    out = model(a)
    print("out.shape is:", out.shape)

