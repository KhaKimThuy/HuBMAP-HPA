from pathlib import Path
class config:
    BASE_PATH = Path("./")
    TRAIN_CSV_PATH = BASE_PATH / 'train.csv'
    TEST_CSV_PATH = BASE_PATH / 'test.csv'
    TRAIN_TIFF_DIR = BASE_PATH / 'train_images'
    TEST_TIFF_DIR = BASE_PATH / 'test_images'
    TRAIN_IMG_DIR = BASE_PATH / 'data/train_png'
    TRAIN_ANNO_DIR = BASE_PATH / 'data/train_anno'
    BATCH_SIZE = 4
    RESOLUTION = 512
    THRESHOLD = 0.5
    TEST_SIZE = 0.2