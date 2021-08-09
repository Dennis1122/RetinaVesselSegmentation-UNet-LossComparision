import os
from utils.augmentations import Train_Generator, Val_Generator


def load_dataset(train_path, test_path):
    train_images_path = os.path.join(train_path, 'images')
    train_mask_path = os.path.join(train_path, 'vessel')

    test_images_path = os.path.join(test_path, 'images')
    test_mask_path = train_mask_path = os.path.join(test_path, 'vessel')

    train_images_files = sorted([os.path.join(train_images_path, i) for i in os.listdir(train_images_path)])
    train_mask_files = sorted([os.path.join(train_mask_path, i) for i in os.listdir(train_mask_path)])

    test_images_files = sorted([os.path.join(test_images_path, i) for i in os.listdir(test_images_path)])
    test_mask_files = sorted([os.path.join(test_mask_path, i) for i in os.listdir(test_mask_path)])

    print("##### Images Loaded #####")
    print("train: ", len(train_images_files), len(train_mask_files))
    print("test: ", len(test_images_files), len(test_mask_files))

    return train_images_files, train_mask_files, test_images_files, test_mask_files


def train_generator(train_images_files, train_mask_files, batch_img_dim):
    return Train_Generator(train_images_files, train_mask_files, batch_size=batch_img_dim[0],
                           img_dim=(batch_img_dim[1], batch_img_dim[2]), augmentation=True).__iter__()


def valid_generator(test_images_files, test_mask_files, batch_img_dim):
    return Val_Generator(test_images_files, test_mask_files, batch_size=batch_img_dim[0],
                         img_dim=(batch_img_dim[1], batch_img_dim[2]), augmentation=True).__iter__()
