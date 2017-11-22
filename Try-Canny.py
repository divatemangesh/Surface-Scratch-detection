import keras
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import mpldatacursor.datacursor
import cv2

from keras.models import Sequential
from keras.layers.core import Dense
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import TensorBoard
plt.ion()


inputIm = cv2.imread("0.png",0)

outputIm = cv2.imread("0_CANNY.png",0)
outputIm.min()

xTrain = []
yTrain = []
kernelSizeHalf = 5
rWise = range(kernelSizeHalf, inputIm.shape[0] - kernelSizeHalf)
cWise = range(kernelSizeHalf, inputIm.shape[1] - kernelSizeHalf)
for r in rWise:
    for c in cWise:
        xBlock = inputIm[r - kernelSizeHalf : r + 1 + kernelSizeHalf, c - kernelSizeHalf : c + 1 + kernelSizeHalf]
        xFlat = xBlock.flatten()
        xTrain.append(xFlat)
        yTrain.append(outputIm[r,c])

training_data = np.array(xTrain)
target_data =np.array(yTrain)
npar=np.array(xTrain)
# training_data.shape
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=32, write_graph=True, write_grads=False,
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                           embeddings_metadata=None)


model = Sequential()
nnInputSize = ((2 * kernelSizeHalf) +1) ** 2            #   input size why this?
model.add(Dense(nnInputSize, input_dim=nnInputSize, activation='relu'))
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])

history=model.fit(training_data, target_data, epochs=5,verbose=2)
print("done history")


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# print("done")
# plt.xlabel('epoch')
# plt.legend(['loss', 'accuracy'], loc='upper left')
# plt.show()

yPredicted = model.predict(training_data)
imPredicted = np.zeros(outputIm.shape)
k = 0
for r in rWise:
    for c in cWise:
        imPredicted[r,c] = yPredicted[k]
        k = k + 1



# imPredicted = 255 * imPredicted / imPredicted.max()
# imPredicted.max()
# plt.imshow(imPredicted)


