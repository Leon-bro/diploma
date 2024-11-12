import deeplabv3PLUS
import tensorflow as tf

model = deeplabv3PLUS.DeeplabV3Plus(1)
predict = model.predict(tf.random.uniform(shape=(1, 224, 224, 1)))
print(predict.shape)
model.summary()

    