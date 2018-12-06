import keras
from sklearn.model_selection import train_test_split
from Load_Datas import Load_Ut_Datas
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# X shape (1000 4096), y shape (1,000, )
X, y = Load_Ut_Datas()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1024)

# 数据预处理
x_train = X_train.astype('float32') / 50. - 0.5  # minmax_normalized
x_test = X_test.astype('float32') / 50. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 压缩特征维度至100维
encoding_dim = 100

# this is our input placeholder
inputs = Input(shape=(4096,))

# 编码层
encoder_output = Dense(encoding_dim, activation='sigmoid', activity_regularizer=keras.regularizers.l1(0.018))(inputs)

# 解码层
decoded = Dense(4096, activation='sigmoid')(encoder_output)

# 构建自编码模型
autoencoder = Model(inputs=inputs, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=inputs, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

autoencoder.summary()

# training
autoencoder.fit(x_train, x_train, epochs=10, batch_size=32, shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()



