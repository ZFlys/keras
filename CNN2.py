from LoadWaveletTransfromDatas import Load_Ut_Datas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt


X, y = Load_Ut_Datas()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1024)

X_train = X_train.reshape(-1, 32, 16, 1)
X_test = X_test.reshape(-1, 32, 16, 1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(7, 5), input_shape=(32, 16, 1)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
    ))
model.add(Conv2D(filters=32, kernel_size=(5, 3), padding='same'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
    ))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.summary()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1)      # validation_data=(X_test, y_test)

score = model.evaluate(X_test, y_test)
print("LOSS:", score[0])
print("Accuracy:", score[1])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

