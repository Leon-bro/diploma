import tensorflow as tf
import pandas as pd
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the learning rate from the optimizer
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = lr(tf.cast(self.model.optimizer.iterations, tf.float32))
        else:
            current_lr = lr
        print(f"Epoch {epoch + 1}: Learning rate is {current_lr.numpy()}")

class LogHistoryToExcel(tf.keras.callbacks.Callback):
    def __init__(self, file_name='training_log.xlsx'):
        super(LogHistoryToExcel, self).__init__()
        self.file_name = file_name
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch + 1
        self.epoch_data.append(logs)

        # Convert epoch data to DataFrame and write to Excel
        df = pd.DataFrame(self.epoch_data)
        df.to_excel(self.file_name, index=False)