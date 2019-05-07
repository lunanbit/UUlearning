import keras
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Conv2D, add, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers, initializers

from keras import regularizers, losses, initializers
from keras.callbacks import Callback, LearningRateScheduler

from dataset import load_dataset
from loss import uu_loss
from helper import generator


class BaseModel():
    def get_data(self):
        U_data1, U_data2, prior_true, x_test, y_test, prior = self.load_data()

        print('Unlabeled training dataset1 shape:', U_data1.shape)
        print('Unlabeled training dataset2 shape:', U_data2.shape)
        print('True data prior', prior_true)
        print('Test data x shape', x_test.shape)
        print('Test data y shape', y_test.shape)
        print('Test data prior', prior)

        return U_data1, U_data2, prior_true, x_test, y_test, prior

    def compile_model(self, model, loss_type, theta1, theta2, prior, mode):
        if self.optimizer is None:
            ValueError("Optimizer is not given.")

        loss_surro = uu_loss(loss_type=loss_type,
                             theta1=theta1,
                             theta2=theta2,
                             prior=prior,
                             mode=mode,
                             surro_flag=True)

        loss_01 = uu_loss(loss_type=loss_type,
                          theta1=theta1,
                          theta2=theta2,
                          prior=prior,
                          mode=mode,
                          surro_flag=False)

        model.compile(loss=loss_surro,
                      optimizer=self.optimizer)
        model.summary()
        self.model = model

    def fit_model(self, U_data1, U_data2, batch_size, epochs, x_test, y_test):
        dim = U_data2.shape[1:]
        shape = np.concatenate(([batch_size], np.array(dim)))
        nb_U1 = U_data1.shape[0]
        nb_U2 = U_data2.shape[0]
        pb_U1 = float(nb_U1 / (nb_U1 + nb_U2))
        nb_batchU1 = int(np.round(batch_size * pb_U1))
        nb_batchU2 = int(batch_size) - nb_batchU1
        if U_data1.shape[0] // nb_batchU1 == U_data2.shape[0] // nb_batchU2:
            steps_per_epoch = U_data1.shape[0] // nb_batchU1
        else:
            steps_per_epoch = np.maximum(U_data1.shape[0] // nb_batchU1, U_data2.shape[0] // nb_batchU2)

        test_loss = TestLoss(self.model, x_test, y_test)

        h = self.model.fit_generator(generator(U_data1, U_data2, pb_U1, batch_size, shape, steps_per_epoch),
                                     steps_per_epoch=steps_per_epoch,
                                     nb_epoch=epochs,
                                     verbose=1,
                                     callbacks=[test_loss])
        loss_test = test_loss.test_losses
        return h.history, loss_test


class MultiLayerPerceptron(BaseModel):
    def __init__(self, dataset, nb_U1, nb_U2, theta1, theta2, mode, loss_type,
                 weight_decay=1e-4):
        self.nb_U1 = nb_U1
        self.nb_U2 = nb_U2
        self.theta1 = theta1
        self.theta2 = theta2
        self.mode = mode
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.dataset = dataset
        self.optimizer = None

    def load_data(self):
        U_data1, U_data2, prior_true, x_test, y_test, prior = load_dataset(self.dataset,
                                                                           self.nb_U1,
                                                                           self.nb_U2,
                                                                           self.theta1,
                                                                           self.theta2,
                                                                           self.mode)

        return U_data1, U_data2, prior_true, x_test, y_test, prior

    def build_model(self, prior, input_shape):
        input = Input(shape=input_shape)

        x = Dense(300, use_bias=False, input_shape=input_shape,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(300, use_bias=False,
                  kernel_initializer=initializers.lecun_normal(seed=1),
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        output = Dense(1, use_bias=True,
                       kernel_initializer=initializers.lecun_normal(seed=1))(x)

        model = Model(inputs=input, outputs=output)

        self.compile_model(model=model,
                           loss_type=self.loss_type,
                           theta1=self.theta1,
                           theta2=self.theta2,
                           prior=prior,
                           mode=self.mode)


class Resnet32Model(BaseModel):
    def __init__(self, dataset, nb_U1, nb_U2, theta1, theta2, mode, loss_type,
                 weight_decay=5e-3):
        self.nb_U1 = nb_U1
        self.nb_U2 = nb_U2
        self.theta1 = theta1
        self.theta2 = theta2
        self.mode = mode
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.dataset = dataset
        self.optimizer = None

    def load_data(self):
        U_data1, U_data2, prior_true, x_test, y_test, prior = load_dataset(self.dataset,
                                                                           self.nb_U1,
                                                                           self.nb_U2,
                                                                           self.theta1,
                                                                           self.theta2,
                                                                           self.mode)

        return U_data1, U_data2, prior_true, x_test, y_test, prior

    def build_model(self, prior, input_shape):
        def residual_network(img_input, classes_num=1, stack_n=5):
            def residual_block(x, o_filters, increase=False):
                stride = (1, 1)
                if increase:
                    stride = (2, 2)

                o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
                conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                                kernel_initializer=initializers.he_normal(seed=1),
                                kernel_regularizer=regularizers.l2(self.weight_decay))(o1)
                o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
                conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                kernel_initializer=initializers.he_normal(seed=1),
                                kernel_regularizer=regularizers.l2(self.weight_decay))(o2)
                if increase:
                    projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                        kernel_initializer=initializers.he_normal(seed=1),
                                        kernel_regularizer=regularizers.l2(self.weight_decay))(o1)
                    block = add([conv_2, projection])
                else:
                    block = add([conv_2, x])
                return block

            # build model ( total layers = stack_n * 3 * 2 + 2 )
            # stack_n = 5 by default, total layers = 32
            # input: 32x32x3 output: 32x32x16
            x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_initializer=initializers.he_normal(seed=1),
                       kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

            # input: 32x32x16 output: 32x32x16
            for _ in range(stack_n):
                x = residual_block(x, 16, False)

            # input: 32x32x16 output: 16x16x32
            x = residual_block(x, 32, True)
            for _ in range(1, stack_n):
                x = residual_block(x, 32, False)

            # input: 16x16x32 output: 8x8x64
            x = residual_block(x, 64, True)
            for _ in range(1, stack_n):
                x = residual_block(x, 64, False)

            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation('relu')(x)
            x = GlobalAveragePooling2D()(x)

            # input: 64 output: 1
            x = Dense(classes_num, kernel_initializer=initializers.he_normal(seed=1))(x)
            return x

        img_input = Input(shape=input_shape)
        output = residual_network(img_input, classes_num=1, stack_n=5)
        model = Model(inputs=img_input, output=output)

        self.compile_model(model=model,
                           loss_type=self.loss_type,
                           theta1=self.theta1,
                           theta2=self.theta2,
                           prior=prior,
                           mode=self.mode)


# Test risk by 01-loss
class TestLoss(Callback):
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        self.test_losses = []

    def on_epoch_end(self, epoch, logs={}):
        # perm = np.random.permutation(len(self.x_test))
        # self.x_test, self.y_test = self.x_test[perm], self.y_test[perm]

        y_test_pred = self.model.predict(self.x_test, batch_size=1000)
        nb_y_test = np.size(self.y_test)

        zero_one_test_loss = np.sum(np.not_equal(np.sign(y_test_pred), np.sign(self.y_test)).astype(np.int32)) / nb_y_test
        print("\n Test loss: %f" % (zero_one_test_loss))
        print("============================================================================")
        self.test_losses.append(zero_one_test_loss)




