import os
import numpy as np
from sklearn import preprocessing


# keras part of tensorflow 2.0
try:
    # keras again imported separately from tensorflow > 2.16
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
except (ModuleNotFoundError, ImportError) as e:
    try:
        # keras part of tensorflow for versions 2.0 to 2.15
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import SGD
    except (ModuleNotFoundError, ImportError) as e:
        # keras part of tensorflow < 2.0
        import keras
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.optimizers import SGD


PERCENTILE = [16, 50, 84]
N_HIDDEN_LAYER_1 = 10
N_HIDDEN_LAYER_2 = 10
N_HIDDEN_LAYER_3 = 5
LEARNING_RATE = 0.02
ACTIVATION = 'tanh'
PATIENCE = 30
DECAY = 1e-6
MOMENTUM = 0.9
INIT_DIST = 'glorot_uniform'


def tilted_loss(n_params, quantiles, y_true, y_pred):
    import tensorflow as tf
    s = len(quantiles)
    q = tf.constant(np.tile(quantiles, n_params), dtype=tf.float32)
    e = tf.concat(
        [
            tf.expand_dims(y_true[:, i], 1) - y_pred[:, i * s: (i + 1) * s]
            for i in range(n_params)
        ],
        axis=1
    )
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.reduce_mean(v)


def apply_network(
        Xdata,
        coeff_file,
        xscale,
        yscale,
        xmean,
        ymean,
        NUMBER_OF_TRUTHS):

    Xtrain_mean = np.asarray([np.loadtxt(xmean)]).astype(np.float32)
    Xtrain_scale = np.asarray([np.loadtxt(xscale)]).astype(np.float32)
    ytrain_mean = np.asarray([np.loadtxt(ymean)]).astype(np.float32)
    ytrain_scale = np.asarray([np.loadtxt(yscale)]).astype(np.float32)
    Xtrain_scale_inv = (1.0 / Xtrain_scale).astype(np.float32)
    Ok_rows = [~np.isnan(Xdata[ind, :]).any() for ind in range(Xdata.shape[0])]
    model = Sequential()
    model.add(keras.Input(shape=(Xdata.shape[1],)))
    model.add(Dense(N_HIDDEN_LAYER_1, activation=ACTIVATION))
    model.add(Dense(N_HIDDEN_LAYER_2, activation=ACTIVATION))
    model.add(Dense(N_HIDDEN_LAYER_3, activation=ACTIVATION))
    model.add(Dense(NUMBER_OF_TRUTHS * len(PERCENTILE), activation='linear'))
    model.load_weights(filepath=coeff_file)
    X_in = (Xdata - Xtrain_mean) * Xtrain_scale_inv
    out = model.predict(X_in[Ok_rows, :], batch_size=int(0.2 * Xdata.shape[0]))
    out = out.reshape(-1, NUMBER_OF_TRUTHS, len(PERCENTILE))
    out = out * ytrain_scale[:, :, np.newaxis] + ytrain_mean[:, :, np.newaxis]
    out_full = np.nan + \
        np.empty((Xdata.shape[0], NUMBER_OF_TRUTHS, len(PERCENTILE)))
    out_full[Ok_rows, :, :] = out
    return out_full


def apply_network_nn_name(
        Xdata,
        NN_NAME="test",
        OUTPUT_DIR=".",
        NUMBER_OF_TRUTHS=5):

    coeff_file = "{:s}/{:s}.keras".format(OUTPUT_DIR, NN_NAME)
    xmean = "{:s}/Xtrain_mean_{:s}.txt".format(OUTPUT_DIR, NN_NAME)
    xscale = "{:s}/Xtrain_scale_{:s}.txt".format(OUTPUT_DIR, NN_NAME)
    ymean = "{:s}/ytrain_mean_{:s}.txt".format(OUTPUT_DIR, NN_NAME)
    yscale = "{:s}/ytrain_scale_{:s}.txt".format(OUTPUT_DIR, NN_NAME)
    return apply_network(
        Xdata,
        coeff_file,
        xscale,
        yscale,
        xmean,
        ymean,
        NUMBER_OF_TRUTHS)


def train_network(
        Xtrain,
        ytrain,
        Xvalid,
        yvalid,
        OUTPUT_DIR=".",
        NN_NAME="test"):
    NUMBER_OF_TRUTHS = ytrain.shape[1]
    # Scale data
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xvalid = scaler.transform(Xvalid)
    scaler2 = preprocessing.StandardScaler().fit(ytrain)
    ytrain = scaler2.transform(ytrain)
    yvalid = scaler2.transform(yvalid)

    scaler_file = os.path.join(OUTPUT_DIR, "{scaler_type}_%s.txt" % (NN_NAME))
    np.savetxt(scaler_file.format(scaler_type="Xtrain_mean"),
               scaler.mean_, delimiter=',')
    np.savetxt(scaler_file.format(scaler_type="Xtrain_scale"),
               scaler.scale_, delimiter=',')
    np.savetxt(scaler_file.format(scaler_type="ytrain_mean"),
               scaler2.mean_, delimiter=',')
    np.savetxt(scaler_file.format(scaler_type="ytrain_scale"),
               scaler2.scale_, delimiter=',')

    np.random.seed(3)

    # Multilayer Perceptron
    Xtrain = Xtrain.astype(np.float32)
    Xvalid = Xvalid.astype(np.float32)
    ytrain = ytrain.astype(np.float32)
    yvalid = yvalid.astype(np.float32)
    # https://en.wikipedia.org/wiki/Quantile_regression
    # https://sachinruk.github.io/blog/Quantile-Regression/

    sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    model = Sequential()

    model.add(Dense(N_HIDDEN_LAYER_1, kernel_initializer=INIT_DIST,
                    activation=ACTIVATION, input_dim=Xtrain.shape[1]))
    model.add(Dense(N_HIDDEN_LAYER_2,
              kernel_initializer=INIT_DIST, activation=ACTIVATION))
    if N_HIDDEN_LAYER_3 > 0:
        model.add(Dense(N_HIDDEN_LAYER_3,
                  kernel_initializer=INIT_DIST, activation=ACTIVATION))
        model.add(Dense(NUMBER_OF_TRUTHS * len(PERCENTILE),
                  kernel_initializer=INIT_DIST, activation='linear'))
    model.compile(loss=lambda y, f: tilted_loss(NUMBER_OF_TRUTHS,
                                                0.01 * np.array(PERCENTILE),
                                                # , optimizer='adagrad')
                                                y, f), optimizer=sgd)

    coeff_file = os.path.join(OUTPUT_DIR, "%s.keras" % (NN_NAME))
    checkpointer = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    checkpointer2 = keras.callbacks.ModelCheckpoint(
        filepath=coeff_file,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    # nnet=model.fit(Xtrain, ytrain, batch_size=250, nb_epoch=2650,
    # validation_datay=(Xvalid,
    # yvalid),callbacks=[checkpointer,checkpointer2],shuffle=True, verbose=2)
    nnet = model.fit(
        Xtrain,
        ytrain,
        batch_size=250,
        epochs=2650,
        validation_data=(
            Xvalid,
            yvalid),
        callbacks=[
            checkpointer,
            checkpointer2],
        shuffle=True,
        verbose=2)

    training_loss = nnet.history['loss']
    validation_loss = nnet.history['val_loss']
    t_loss_file = os.path.join(OUTPUT_DIR, "%s_tloss.txt" % (NN_NAME))
    v_loss_file = os.path.join(OUTPUT_DIR, "%s_vloss.txt" % (NN_NAME))
    np.savetxt(t_loss_file, training_loss, delimiter=',')
    np.savetxt(v_loss_file, validation_loss, delimiter=',')
    print(min(training_loss))
    print(min(validation_loss))

    # epoch at which validation loss is minimum
    validation_loss.index(min(validation_loss))
    [training_loss[i]
     for i in (np.where(validation_loss == min(validation_loss))[0]).tolist()]
    min(validation_loss)


if __name__ == '__main__':
    pass
