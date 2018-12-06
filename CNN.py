from LoadWaveletTransfromDatas import Load_Ut_Datas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras


X, y = Load_Ut_Datas()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(512,)))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=1200, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test)
print("LOSS:", score[0])
print("Accuracy::", score[1])
