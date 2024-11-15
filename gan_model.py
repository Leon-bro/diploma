import tensorflow as tf
import numpy as np
import time
import os
OUTPUT_CHANNELS = 1

class Downsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)    
        x1 = self.down1(x, training=training) # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training) # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training) # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training) # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training) # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training) # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training) # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training) # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training) # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training) # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training) # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training) # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training) # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training) # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training) # (bs, 128, 128, 128)

        x16 = self.last(x15) # (bs, 256, 256, 3)
        x16 = tf.nn.sigmoid(x16)

        return x16

class DiscDownsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        # we are zero padding here with 1 because we need our shape to 
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1) # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training) # (bs, 128, 128, 64)
        x = self.down2(x, training=training) # (bs, 64, 64, 128)
        x = self.down3(x, training=training) # (bs, 32, 32, 256)

        x = self.zero_pad1(x) # (bs, 34, 34, 256)
        x = self.conv(x)      # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x) # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)      # (bs, cen, 30, 1 )

        return x
class GANModel:
    def __init__(self, lambda_value=100, checkpoint_dir='checkpoints'):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lambda_value = lambda_value
        self.gen_optimizer = tf.optimizers.Adam(beta_1=0.5, epsilon=2e-4)
        self.disc_optimizer = tf.optimizers.Adam(beta_1=0.5, epsilon=2e-4)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = []

        # Checkpoint to save/restore only the generator
        self.checkpoint = tf.train.Checkpoint(generator=self.generator, gen_optimizer=self.gen_optimizer)
    def summary(self):
        if self.generator is not None:
            print(self.generator.summary())
        if self.discriminator is not None:
            print(self.discriminator.summary())
        print("Total params: ", self.generator.count_params() + self.discriminator.count_params())
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_fn(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_fn(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        # GAN loss using discriminator's feedback on generated output
        gan_loss = self.loss_fn(tf.ones_like(disc_generated_output), disc_generated_output)

        # L1 loss for pixel-wise accuracy in binary segmentation
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # Total generator loss
        return gan_loss + (self.lambda_value * l1_loss)

    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate an image using the generator
            gen_output = self.generator(input_image, training=True)

            # Get discriminator's classification for real and generated images
            disc_real_output = self.discriminator(input_image, target, training=True)
            disc_generated_output = self.discriminator(input_image, gen_output, training=True)

            # Calculate losses
            gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        # Apply gradients to the generator and discriminator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, epochs, validation_data=None, save_period=5):
        for epoch in range(epochs):
            start = time.time()
            for input_image, target in dataset:
                gen_loss, disc_loss = self.train_step(input_image, target)

            # Save the model checkpoint at the end of each epoch
            if epoch % save_period == 0:
                self.save_model()

            # Evaluate the generator on the validation data, if provided
            print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}, '
                      f'Time: {time.time() - start}s\n')
            tf.print('Train evaluate:')
            self.evaluate(dataset)
            tf.print('validation evaluate:')
            self.evaluate(validation_data)

    def save_model(self):
        """Saves the generator model and optimizer state."""
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print("Generator model saved.")
    def compile(self, metrics):
        self.metrics = metrics
    def load_model(self):
        """Loads the generator model and optimizer state."""
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest:
            self.checkpoint.restore(latest)
            print("Generator model restored from", latest)
        else:
            print("No checkpoint found. Starting from scratch.")
    def evaluate(self, input):
        values = []
        for batch in input:
            im, lb = batch
            values.append(self._evaluate_step(im, lb))
        values = tf.reduce_mean(values, axis=0)
        print(",".join([m.name + '=' + str(v) for m, v in zip(self.metrics, values.numpy())]))

    def _evaluate_step(self, input_image, target):
        """Generates output for a given input image using the generator."""
        prediction = self.generator(input_image, training=False)
        values = []
        for metric in self.metrics:
            values.append(metric(prediction, target))

        return values
