import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    # env2cluster = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    cluster_number = 3
    epoch = 87
    X = []
    Y = []
    for i in range(epoch):
        sample = np.load('model/colight_maml_cluster_new2/samples_{}.npy'.format(i))
        env2cluster = np.load('model/colight_maml_cluster_new2/env2cluster_{}.npy'.format(i))
        for j in range(len(sample)):
            s = []
            for k in range(len(sample[j])):
                if k < 100:
                    s.append(np.ravel(sample[j][k]))
            X.append(s)
            Y.append(env2cluster[j])
    X = np.array(X)
    Y = np.array(Y)
    Y = keras.utils.to_categorical(Y, num_classes=cluster_number)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    model = Sequential()
    # model.add(LSTM(units=32, input_shape=(360, 6, 16)))
    model.add(LSTM(units=128, activation='tanh'))
    # model.add(LSTM(units=32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(cluster_number, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=100, verbose=2)
    result = model.predict(X_test)
    name = "model/colight_maml_cluster_new2/tailor.h5"
    model.save_weights(name)
    model_json = model.to_json()
    with open('model/colight_maml_cluster_new2/tailor.json', 'w') as file:
        file.write(model_json)
    print('hello world')