from keras.losses import binary_crossentropy
from keras import backend as K

# Dice coefficient đo lường mức độ trùng hợp giữa hai vùng được dự đoán và thực tế.
# Dice coefficient được tính bằng cách tính tỉ lệ giữa diện tích hai vùng (vùng được
# dự đoán và vùng thực tế) và diện tích tổng của cả hai vùng. Giá trị của Dice coefficient
# nằm trong khoảng từ 0 đến 1, với giá trị 0 thể hiện hoàn toàn không trùng hợp và giá trị
# 1 thể hiện hoàn toàn trùng hợp giữa hai vùng.
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_bce_loss(y_true, y_pred):
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    bce_loss = binary_crossentropy(y_true, y_pred)
    return dice_loss + bce_loss