# source of help: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("finish loading\n")

y = train["label"]
X = train.drop(columns="label")

# normalization
X = X / 255.0
test = test / 255.0

# Reshape
X = X.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
# where is this "values" from? what is the parameter -1?

print(X.shape)
print("\n")

# convert values in y into hot vectors
Y = to_categorical(y, num_classes=10)

# cross validation. split the training data into training set and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state=2)


# build the model
# the tutorial used sequential model in keras so we only need to add layers into it
# here try to do a simple model using method introduced by deeplearning.ai in coursera's CNN course

# really easy model, only one Conv layer, one maxpooling layer, one fc layer
def mymodel(input_shape):
    X_input = Input(input_shape)

    # 32 filters, shape of the kernel is (4,4), stride 1, same padding
    X = Conv2D(32, (4, 4), padding='Same', name='conv0')(X_input)

    # Q: what's the aim of batch normalization?
    # A: to make the outcome more sensitive to activation function
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = Conv2D(32, (4, 4), padding='Same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPool2D((2, 2), name='max_pool1')(X)

    # use dropout to avoid overfit
    # Q: why 0.25?
    X = Dropout(0.25)(X)

    X = Conv2D(64, (3, 3), padding='Same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(64, (3, 3), padding='Same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = MaxPool2D((2, 2), strides=(2, 2), name='max_pool2')(X)
    X = Dropout(0.25)(X)

    # TODO
    # what's Flatten() for?
    X = Flatten()(X)
    # X = Dense(256, activation='relu')(X)
    # why 0.5 drop out rate here?
    # X = Dropout(0.5)(X)
    X = Dense(10, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='mymodel')
    return model


# declare the model
themodel = mymodel((28, 28, 1))

# define the optimizer(values are all default values)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# define the learning rate reduce. reduce the learning rate by half when the accuracy didn't improved for 3 epochs.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# compile the model
themodel.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# fit the model now
themodel.fit(X_train, Y_train, batch_size=30, epochs=86, callbacks=[learning_rate_reduction])

preds = themodel.evaluate(x=X_val, y=Y_val)

print("Loss: " + str(preds[0]))
print("Accuracy: " + str(preds[1]))

result = themodel.predict(test)
result = np.argmax(result, axis=1)
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), pd.Series(result, name="Label")], axis=1)
submission.to_csv("SingleConvNet2.csv", index=False)
