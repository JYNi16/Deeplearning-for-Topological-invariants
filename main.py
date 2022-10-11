import tensorflow as tf
from tensorflow import keras
from model import *
from keras import layers, Model, losses
from keras import layers, optimizers, metrics
from dataloader import *

optimizer = optimizers.Adam(lr=0.001)
acc_meter = metrics.Accuracy()

def train(train_data):
    model = CNN_TI()
    loss_all = 0
    for epoch in range(25):
        for step, data in enumerate(train_data):
            with tf.GradientTape() as tape:
                x, y = data
                x = tf.reshape(x, [-1, 33, 2, 1])
                out = model(x)
                y_onehot = tf.one_hot(y, depth=3)
                #loss = tf.square(out - y_onehot)
                #loss = tf.reduce_sum(loss) / config.batch_size
                loss = tf.keras.losses.categorical_crossentropy(from_logits=True, y_true=y_onehot, y_pred=out)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                acc_meter.update_state(tf.argmax(out, axis=1), y)
            if (step % 10 == 0):
                print("step is:", step)
        print("epoch:", epoch, "| Loss is:%.4f" % (loss), "|Accuracy is:%.4f" % (acc_meter.result().numpy()))
        acc_meter.reset_states()
    #tf.saved_model(model, "saved/1")

if __name__=="__main__":
    path = "E:/deeplearning\depp learning for Phys/Deep-Learning-Topological-Invariants/train_data1"
    data = HamData(path)
    db_train = data.createdata()
    train(db_train)