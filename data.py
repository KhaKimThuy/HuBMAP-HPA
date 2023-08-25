import os
from pathlib import Path
from tensorflow import keras
from utils import *

class HubmapOrgan(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, shuffle=True, augment=None):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.input_img_paths) / self.batch_size)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.input_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1))
        for j, img_path in enumerate(batch_img_paths):

            x[j] = preprocess_image(img_path, enhance=True)

            image_id = Path(img_path).stem
            mask_path = os.path.join(train_anno_dir, f'{image_id}.png')

            original_mask = keras.utils.load_img(mask_path, color_mode="grayscale")
            mask = original_mask.resize((self.img_size, self.img_size))
            mask_array = keras.utils.img_to_array(mask)
            y[j] = mask_array

        if self.augment is None:
            return x.astype('float32'), y.astype('float32')/255
        else:
            aug_x, aug_y = [], []
            for image, mask in zip(x, y):
                transformed = self.augment(image=image, mask=mask)
                aug_x.append(transformed['image'])
                aug_y.append(transformed['mask'])
            return np.array(aug_x), np.array(aug_y)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.input_img_paths)