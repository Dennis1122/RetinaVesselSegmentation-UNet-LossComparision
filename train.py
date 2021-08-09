import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint
from utils import dataset_utils as ds
from utils.augmentations import Train_Generator, Val_Generator
from models.unet import build_unet
from utils.metrics import iou, dice
import config

# Global variables
train_path = config.train_path
test_path = config.test_path
H = config.img_height  # height of image
W = config.img_width  # width of image
batch_img_dim = config.batch_img_dim
batch_msk_dim = config.batch_msk_dim

# Training Configuration
EXP_NAME = config.EXP_NAME
loss_function = config.loss_function


def train_model():
    # load dataset
    train_images_files, train_mask_files, test_images_files, test_mask_files = ds.load_dataset(train_path, test_path)

    def train_generator():
        return Train_Generator(train_images_files, train_mask_files, batch_size=batch_img_dim[0],
                               img_dim=(batch_img_dim[1], batch_img_dim[2]), augmentation=True).__iter__()

    def valid_generator():
        return Val_Generator(test_images_files, test_mask_files, batch_size=batch_img_dim[0],
                             img_dim=(batch_img_dim[1], batch_img_dim[2]), augmentation=True).__iter__()

    # create test and train dataset
    ds_train = tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_img_dim[0], batch_img_dim[1], batch_img_dim[2], batch_img_dim[3]],
                       [batch_msk_dim[0], batch_msk_dim[1], batch_msk_dim[2], batch_msk_dim[3]])
    ).repeat()

    ds_valid = tf.data.Dataset.from_generator(
        valid_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_img_dim[0], batch_img_dim[1], batch_img_dim[2], batch_img_dim[3]],
                       [batch_msk_dim[0], batch_msk_dim[1], batch_msk_dim[2], batch_msk_dim[3]])
    ).repeat()

    # Experiment folder setup
    training_folder = os.path.join(EXP_NAME, "training")
    model_path = os.path.join(training_folder, "model.h5")
    csv_path = os.path.join(training_folder, "data.csv")
    logs_dir = f"{training_folder}/logs"

    # Build model
    model = build_unet((H, W, 3))

    # Compile model
    model.compile(optimizer=Adam(), loss=loss_function, metrics=[iou, dice])

    # Define Callbacks
    earlystopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        restore_best_weights=True)
    reducelr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        min_delta=0.0001)
    csv_logger = CSVLogger(csv_path)
    tensor_board = TensorBoard(log_dir=logs_dir, histogram_freq=1)
    model_ckpt = ModelCheckpoint(model_path, verbose=1, save_best_only=True)

    callbacks = [earlystopping, reducelr, csv_logger, tensor_board, model_ckpt]

    # train the model
    history = model.fit(
        ds_train,
        steps_per_epoch=40,
        epochs=100,
        validation_data=ds_valid,
        validation_steps=4,
        callbacks=callbacks
    )


if __name__ == '__main__':
    train_model()
