import argparse
import os
from pathlib import Path

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from keras_noise2noise.generator import NoisyImageGenerator, ValGenerator
from keras_noise2noise.model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from keras_noise2noise.noise_model import get_noise_model
from keras_noise2noise.train import Schedule

from resolution_enhancer.settings import MODELS_DIR, DATA_DIR, MODEL_ARCHITECTURE


def get_args():
    parser = argparse.ArgumentParser(
        description="train noise2noise model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        help="train image dir",
        default=str(DATA_DIR / "resolution_enhancer_dataset" / "training"),
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        help="test image dir",
        default=str(DATA_DIR / "resolution_enhancer_dataset" / "validation"),
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="training patch size"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="steps per epoch")
    parser.add_argument(
        "--loss", type=str, default="mse", help="loss; mse', 'mae', or 'l0' is expected"
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="weight file for restart"
    )
    parser.add_argument(
        "--output_path", type=str, default=str(MODELS_DIR), help="checkpoint dir"
    )
    parser.add_argument(
        "--source_noise_model",
        type=str,
        default="clean",
        help="noise model for source images",
    )
    parser.add_argument(
        "--target_noise_model",
        type=str,
        default="clean",
        help="noise model for target images",
    )
    parser.add_argument(
        "--val_noise_model",
        type=str,
        default="clean",
        help="noise model for validation source images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ARCHITECTURE,
        help="model architecture ('srresnet' or 'unet')",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    image_dir = args.image_dir
    test_dir = args.test_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    output_path = Path(args.output_path)
    model = get_model(args.model)

    if args.weight is not None:
        model.load_weights(args.weight)

    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    source_noise_model = get_noise_model(args.source_noise_model)
    target_noise_model = get_noise_model(args.target_noise_model)
    val_noise_model = get_noise_model(args.val_noise_model)
    input_image_dir = os.path.join(image_dir, "input_images")
    target_image_dir = os.path.join(image_dir, "target_images")
    generator = NoisyImageGenerator(
        input_image_dir,
        target_image_dir,
        source_noise_model,
        target_noise_model,
        batch_size=batch_size,
        image_size=image_size,
    )
    validation_input_image_dir = os.path.join(test_dir, "input_images")
    validation_target_image_dir = os.path.join(test_dir, "target_images")
    val_generator = ValGenerator(
        validation_input_image_dir, validation_target_image_dir, val_noise_model
    )
    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(
        ModelCheckpoint(
            os.path.join(
                output_path, "resolution_enhancer_{}.h5".format(MODEL_ARCHITECTURE)
            ),
            monitor="val_PSNR",
            verbose=1,
            mode="max",
            save_best_only=True,
        )
    )

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps,
        epochs=nb_epochs,
        validation_data=val_generator,
        verbose=1,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
