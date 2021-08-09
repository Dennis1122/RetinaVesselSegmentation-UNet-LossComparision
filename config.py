from utils.loss_functions import binary_crossentropy, dice_loss, cldice_loss, \
    combined_cldice_loss, tversky_loss, focal_tversky

EXP_NAME = "exp_binary_crossentropy"
loss_function = binary_crossentropy
train_path = "dataset/DRIVE/training"
test_path = "dataset/DRIVE/test"
img_height = 512
img_width = 512
batch_img_dim = (8, 512, 512, 3)
batch_msk_dim = (8, 512, 512, 1)
