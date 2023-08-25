import os
from PIL import Image
import cv2
import numpy as np
import tifffile as tiff
import csv

def rle2mask(mask_rle, shape=(1600,256)): # decoder
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def split_image(df, img_folder, mask_folder):

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    with open('check.csv', 'w', newline='') as csv_file:

        dw = csv.DictWriter(csv_file, delimiter=',', fieldnames=['id', 'organ', 'ratio'])
        dw.writeheader()

        writer = csv.writer(csv_file)

        for id_ in df.id:
            img = tiff.imread(f'{config.TRAIN_IMAGES_PATH}\{id_}.tiff')
            mask = rle2mask(df[df["id"]==id_]["rle"].values[0], (img.shape[1], img.shape[0]))
            organ = df.loc[df.id==id_]['organ'].values[0]

            w, h = img.shape[:2]
            img1 = img[:int(w/2),:int(h/2)]
            img2 = img[:int(w/2),int(h/2):]
            img3 = img[int(w/2):,:int(h/2)]
            img4 = img[int(w/2):,int(h/2):]

            m1 = mask[:int(w/2),:int(h/2)]
            m2 = mask[:int(w/2),int(h/2):]
            m3 = mask[int(w/2):,:int(h/2)]
            m4 = mask[int(w/2):,int(h/2):]

            r1 = np.count_nonzero(m1 == 1)/img1.size
            r2 = np.count_nonzero(m2 == 1)/img1.size
            r3 = np.count_nonzero(m3 == 1)/img1.size
            r4 = np.count_nonzero(m4 == 1)/img1.size

            # Save image
            tiff.imsave(f'{img_folder}/{id_}_1.tiff', img1)
            tiff.imsave(f'{img_folder}/{id_}_2.tiff', img2)
            tiff.imsave(f'{img_folder}/{id_}_3.tiff', img3)
            tiff.imsave(f'{img_folder}/{id_}_4.tiff', img4)
            tiff.imsave(f'{img_folder}/{id_}.tiff', img)
            

            # Save mask
            plt.imsave(f'{mask_folder}/{id_}_1.png', np.array(m1), cmap='gray')
            plt.imsave(f'{mask_folder}/{id_}_2.png', np.array(m2), cmap='gray')
            plt.imsave(f'{mask_folder}/{id_}_3.png', np.array(m3), cmap='gray')
            plt.imsave(f'{mask_folder}/{id_}_4.png', np.array(m4), cmap='gray')
            plt.imsave(f'{mask_folder}/{id_}.png', np.array(mask), cmap='gray')

            # Backup for info
            writer.writerow([f'{id_}_1', organ, r1])
            writer.writerow([f'{id_}_2', organ, r2])
            writer.writerow([f'{id_}_3', organ, r3])
            writer.writerow([f'{id_}_4', organ, r4])

def tiff_2_png(tiff_folder_path, folder_to_save):
    for i in os.listdir(tiff_folder_path):
        image = Image.open(f'{tiff_folder_path}/{i}')
        img_name = i.split('.')[0]
        image.save(f'{folder_to_save}/{img_name}.png')
        os.remove(f'{tiff_folder_path}/{i}')

def enhance_image(bgr_img):
    lab= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(12,12))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def preprocess_image(path, enhance=False):
    image = cv2.imread(path)
    if enhance:
        image = enhance_image(image)
    image = cv2.resize(image, (img_size, img_size))
    image_array = image/255
    return image_array
