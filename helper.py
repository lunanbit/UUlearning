import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generator(x1, x2, prob, nb_batch, shape, steps_per_epoch):
    n_x1 = x1.shape[0]
    n_x2 = x2.shape[0]
    n_batch1 = int(np.round(nb_batch*prob))
    n_batch2 = int(nb_batch) - n_batch1
    # assert ((n_x1//n_batch1) == (n_x2//n_batch2))

    start1 = 0
    start2 = 0
    epoch_finished = False
    iter = 0
    while True:
        if epoch_finished:
            np.random.shuffle(x1)
            np.random.shuffle(x2)
            start1 = 0
            start2 = 0
            epoch_finished = False
            iter = 0
        else:
            batch_x = np.zeros(shape=shape)
            stop1 = start1 + n_batch1
            diff1 = stop1 - n_x1
            stop2 = start2 + n_batch2
            diff2 = stop2 - n_x2
            if diff1 <= 0:
                batch1 = x1[start1:stop1]
                start1 += n_batch1
            elif 0 < diff1 < n_batch1:
                batch1 = np.concatenate((x1[start1:], x1[:diff1]), axis=0)
                start1 = diff1
            else:
                start1_ = start1 % n_x1
                batch1 = x1[start1_:diff1]
                start1 = diff1
            batch_x[:n_batch1, ] = batch1
            if diff2 <= 0:
                batch2 = x2[start2:stop2]
                start2 += n_batch2
            elif 0 < diff2 < n_batch2:
                batch2 = np.concatenate((x2[start2:], x2[:diff2]), axis=0)
                start2 = diff2
            else:
                start2_ = start2 % n_x2
                batch2 = x2[start2_:diff2]
                start2 = diff2
            batch_x[n_batch1:nb_batch, ] = batch2
            batch_x = np.asarray(batch_x, dtype=np.float32)
            batch_y = np.concatenate((np.ones((n_batch1, 1), dtype=np.int32),
                                      -np.ones((n_batch2, 1), dtype=np.int32)), axis=0)
            assert (len(batch_x) == len(batch_y))
            perm = np.random.permutation(len(batch_x))
            batch_x, batch_y = batch_x[perm], batch_y[perm]
            iter += 1
            if iter == steps_per_epoch:
                epoch_finished = True
            yield batch_x, batch_y


def plot_loss(np_loss_test, nb_epoch):
    plt.xlim(1, nb_epoch, 1)
    plt.plot(range(1, nb_epoch+1), np_loss_test, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)


def lr_decay_schedule(loss, theta1, theta2):
    if loss == 'sigmoid':
        if 1 - theta1 + theta2 < 0.2:
            lr_decay = 0
        elif 0.2 < 1 - theta1 + theta2 <= 0.3:
            lr_decay = 1e-6
        elif 0.3 < 1 - theta1 + theta2 <= 0.4:
            lr_decay = 1e-4
        else:
            lr_decay = 1e-3
    else:
        if 1 - theta1 + theta2 < 0.2:
            lr_decay = 0
        elif 0.2 < 1 - theta1 + theta2 <= 0.3:
            lr_decay = 5e-6
        elif 0.3 < 1 - theta1 + theta2 <= 0.4:
            lr_decay = 5e-5
        else:
            lr_decay = 5e-4
    return lr_decay
