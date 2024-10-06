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

class LinearDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=1E-4, final_lr=1E-6, for_epochs=200):
        super(LinearDecayScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = for_epochs
        self.step = (self.initial_lr - self.final_lr) / self.total_epochs
        print(self.step)

    # Function to compute the learning rate for each epoch
    def linear_decay(self, epoch):
        if epoch <= self.total_epochs:
            self.initial_lr -= self.step
            return self.initial_lr
        else:
            return self.final_lr

    # Overriding on_epoch_begin to set the learning rate before each epoch
    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.linear_decay(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print(f"Epoch {epoch+1}: Setting learning rate to {new_lr:.10f}")