import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist= tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()

# Normalize the data and reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# Build a CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#
model.fit(x_train,y_train,epochs=3)

model.save('handwritten.keras')

model=tf.keras.models.load_model('handwritten.keras')

image_number=1
while os.path.isfile(f"img/digit{image_number}.png"):
    try:
        img=cv2.imread(f"img/digit{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = img.reshape(1, 28, 28) / 255.0  # Normalize and reshape
        prediction=model.predict(img)
        print(f"this digit is probably a{np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error!")
    finally:
        image_number+=1



