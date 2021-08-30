import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf


class DigitClassifier:
    DATA_ROWS = DATA_COLS = 28
    DATA_CLASSES = 10

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.fetch_data()
        self.model = self.model_gen()

    def fetch_data(self):
        """
        Fetch and normalize
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], self.DATA_ROWS, self.DATA_COLS, 1) / 255
        x_test = x_test.reshape(x_test.shape[0], self.DATA_ROWS, self.DATA_COLS, 1) / 255
        return (x_train, y_train), (x_test, y_test)

    def model_gen(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation=tf.keras.activations.relu,
                                   input_shape=(self.DATA_ROWS, self.DATA_COLS, 1)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128,
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(units=self.DATA_CLASSES,
                                  activation=tf.keras.activations.softmax)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model

    def train(self):
        epochs = 10
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(), "weights"),
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            verbose=1,
            validation_data=(self.x_test, self.y_test),
            callbacks=[cp_callback]
        )
        return history


dc = DigitClassifier()

