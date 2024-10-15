from tensorflow.keras import backend as K
import tensorflow as tf
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=1):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
                    tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result
    
class BinaryCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, name='binary_crossentropy_loss',beta=2, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__(name=name, **kwargs)
        self.beta = beta
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Щоб уникнути log(0)

        loss = -(self.beta * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)
    
class CombainedDiceBinaryLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.9):
        super(CombainedDiceBinaryLoss, self).__init__()
        self.name = 'CombainedDiceBinaryLoss'
        self.alpha = alpha
    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        binary_loss = BinaryCrossEntropyLoss()(y_true, y_pred)
        dice_loss = DiceLoss()(y_true, y_pred)
        combined_loss = binary_loss + self.alpha*dice_loss
        return combined_loss