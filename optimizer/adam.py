import numpy as np
from utils.cnn_utils import get_minibatches, accuracy
from tqdm import tqdm
import time


class Adam(object):

    def __init__(self, model, X_train, y_train,
                 learning_rate, epoch, minibatch_size, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.epoch = epoch
        self.X_test = X_test
        self.y_test = y_test
        self.num_layer = len(self.model.layers)
        self.minibatch_size = minibatch_size

    def initialize_adam(self):
        VdW, Vdb, SdW, Sdb = [], [], [], []
        for param_layer in self.model.params:
            if len(param_layer) is not 2:  # layers which has no learning
                VdW.append(np.zeros_like([]))
                Vdb.append(np.zeros_like([]))
                SdW.append(np.zeros_like([]))
                Sdb.append(np.zeros_like([]))
            else:
                VdW.append(np.zeros_like(param_layer[0]))
                Vdb.append(np.zeros_like(param_layer[1]))
                SdW.append(np.zeros_like(param_layer[0]))
                Sdb.append(np.zeros_like(param_layer[1]))

        assert len(VdW) == self.num_layer
        assert len(Vdb) == self.num_layer
        assert len(SdW) == self.num_layer
        assert len(Sdb) == self.num_layer

        return VdW, Vdb, SdW, Sdb

    def update_parameters(self, VdW, Vdb, SdW, Sdb, grads, t):

        VdW_corrected = [np.zeros_like(v) for v in VdW]
        Vdb_corrected = [np.zeros_like(v) for v in Vdb]
        SdW_corrected = [np.zeros_like(s) for s in SdW]
        Sdb_corrected = [np.zeros_like(s) for s in Sdb]

        # compute dW, db using current mini batch

        grads = list(reversed(grads))
        for i in range(len(grads)):
            if len(grads[i]) is not 0:   # layer which contains weights and biases

                # Moving average of the gradients (Momentum)

                a = self.beta1 * VdW[i]
                b = (1 - self.beta1) * grads[i][0]
                VdW[i] = np.add(a, b)

                a = self.beta1 * Vdb[i]
                b = (1 - self.beta1) * grads[i][1]
                Vdb[i] = np.add(a, b)

                # Moving average of the squared gradients. (RMSprop)
                a = self.beta2 * SdW[i]
                b = (1-self.beta2) * np.power(grads[i][0], 2)
                SdW[i] = np.add(a, b)

                a = self.beta2 * Sdb[i]
                b = (1-self.beta2) * np.power(grads[i][1], 2)
                Sdb[i] = np.add(a, b)

                # Compute bias-corrected first moment estimate

                den = (1-(self.beta1 ** t))
                VdW_corrected[i] = np.divide(VdW[i], den)
                Vdb_corrected[i] = np.divide(Vdb[i], den)

                # Compute bias-corrected second raw moment estimate
                den = 1-(self.beta2 ** t)
                SdW_corrected[i] = np.divide(SdW[i], den)
                Sdb_corrected[i] = np.divide(Sdb[i], den)

                # weight update
                den = np.sqrt(SdW_corrected[i]) + self.epsilon
                self.model.params[i][0] = self.model.params[i][0] - self.learning_rate * np.divide(VdW_corrected[i], den)

                # bias update
                den = np.sqrt(Sdb_corrected[i]) + self.epsilon
                self.model.params[i][1] = self.model.params[i][1] - self.learning_rate * np.divide(Vdb_corrected[i], den)

    def minimize(self):
        costs = []
        t = 0
        np.random.seed(1)
        VdW, Vdb, SdW, Sdb = self.initialize_adam()
        for i in tqdm(range(self.epoch)):
            start = time.time()
            loss = 0
            minibatches = get_minibatches(self.X_train,
                                          self.y_train,
                                          self.minibatch_size)
            for minibatch in tqdm(minibatches):
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # forward and backward propogation
                loss, grads = self.model.fit(minibatch_X, minibatch_Y)
                loss += loss
                t = t + 1  # Adam counter
                # weight update
                self.update_parameters(VdW, Vdb, SdW, Sdb, grads, t)

            # Print the cost every epoch
            end = time.time()
            epoch_time = end - start
            train_acc = accuracy(self.model.predict(self.X_train),
                                 self.y_train)
            val_acc = accuracy(self.model.predict(self.X_test),
                               self.y_test)
            print ("Cost after epoch %i: %f" % (i, loss),
                   'time (s):', epoch_time,
                   'train_acc:', train_acc,
                   'val_acc:', val_acc)
            costs.append(loss)
        print ('total_cost', costs)

        return self.model, costs
