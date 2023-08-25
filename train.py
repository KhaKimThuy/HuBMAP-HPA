import os
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from loss import dice_coefficient, dice_bce_loss
from model import conv_block, upsampling_block, unet_model
from get_config import config
from utils import *


# Data preparation
data_folder = str(config.TRAIN_IMG_DIR).split('\\')[0]
os.makedirs(data_folder, exist_ok=True)

data = pd.read_csv(config.TRAIN_CSV_PATH)
split_image(data, config.TRAIN_IMG_DIR, config.TRAIN_ANNO_DIR)
tiff_2_png(config.TRAIN_IMG_DIR, config.TRAIN_IMG_DIR)

input_img_paths = sorted([os.path.join(TRAIN_IMG_DIR, f"{id}") for id in os.listdir(config.TRAIN_IMG_DIR)])
random.Random(1337).shuffle(input_img_paths)

# Train - val split
val_num = int(len(input_img_paths)*config.TEST_SIZE)
train_img_paths = input_img_paths[:-val_num]
val_img_paths = input_img_paths[-val_num:]

train_gen = HubmapOrgan(config.BATCH_SIZE, config.RESOLUTION, train_img_paths)
val_gen = HubmapOrgan(config.BATCH_SIZE, config.RESOLUTION, val_img_paths)

# Model
model = unet_model(input_size=(img_size,img_size,3))
model_checkpoint=tf.keras.callbacks.ModelCheckpoint('/kaggle/working/epoch_{epoch:02d}.h5',period=2,save_weights_only=False)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=dice_bce_loss,
              metrics=[dice_coefficient])

# Train
model.fit(train_gen, epochs=300, validation_data=val_gen, callbacks=[model_checkpoint],shuffle=True)
model.save('final.h5')