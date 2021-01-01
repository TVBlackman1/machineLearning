from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train /= 255

y_train = utils.to_categorical(y_train, 10)

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

model = Sequential()

model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])

print(model.summary())
