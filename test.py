from PIL import Image
import cv2
from get_config import config
import matplotlib.pyplot as plt
from tensorflow import keras


# Load model
model = keras.models.load_model('best.h5', compile=False)

# Image id in TRAIN set
ids = ['1229', '2079', '3054', '8227_3', '9445_2']

rows = 5
cols = 2
plt.figure(figsize=((cols*5, rows*5)))

for i in range(0, rows*cols, 2):

    path = f'{config.TRAIN_IMG_DIR}/{ids[int(i/2)]}.png'
    mask = f'{config.TRAIN_ANNO_DIR}/{ids[int(i/2)]}.png'

    mask = cv2.imread(mask, 0)
    mask = cv2.resize(mask, (config.RESOLUTION, config.RESOLUTION))
    image = preprocess_image(path, enhance=True)
    pred = model.predict(np.expand_dims(image, axis=0))
    pred_mask = np.where(pred > config.THRESHOLD, 1, 0)[0]

    plt.subplot(rows, cols, i+1)
    plt.title('Ground truth')
    plt.imshow(image)
    plt.imshow(mask, cmap='seismic', alpha=0.4)

    plt.subplot(rows, cols, i+2)
    plt.title('Predicted')
    plt.imshow(image)
    plt.imshow(pred_mask, cmap='seismic', alpha=0.4)
