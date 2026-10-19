from torchvision import transforms
import random


def configure_augmentations(config, train_transforms, resize_param):
    # This function configures only augmentation-specific transforms based on the provided configuration.
    if bool(config.augmentation.on):
        if bool(config.augmentation.crop):
            train_transforms.append(transforms.RandomApply(
                [transforms.RandomResizedCrop(size=(resize_param, resize_param),
                                              scale=config.augmentation.crop_scale)],
                p=config.augmentation.crop_probs))
        if bool(config.augmentation.horizontal_flip):
            train_transforms.append(transforms.RandomHorizontalFlip(p=config.augmentation.hf_probs))
        if bool(config.augmentation.vertical_flip):
            train_transforms.append(transforms.RandomVerticalFlip(p=config.augmentation.vf_probs))
        if bool(config.augmentation.color_jitter):
            train_transforms.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=config.augmentation.brightness,
                                        contrast=config.augmentation.contrast,
                                        saturation=config.augmentation.saturation,
                                        hue=config.augmentation.hue)],
                p=config.augmentation.jitter_probs))
        if bool(config.augmentation.adjust_sharpness):
            train_transforms.append(
                transforms.RandomAdjustSharpness(sharpness_factor=config.augmentation.sharpness_factor,
                                                 p=config.augmentation.sharpness_probs))
        if bool(config.augmentation.blur):
            train_transforms.append(transforms.RandomApply(
                [transforms.GaussianBlur(config.augmentation.window,
                                         sigma=config.augmentation.sigma)],
                p=config.augmentation.blur_probs))

    return train_transforms


def setup_augmentation(config):
    resize_param = config.augmentation.resize_param
    train_transforms = []
    val_transforms = []

    # Common resize step for both training and validation
    train_transforms.append(transforms.Resize((resize_param, resize_param)))
    val_transforms.append(transforms.Resize((resize_param, resize_param)))

    if bool(config.augmentation.on):
        # Configure augmentations specifically for the training set
        augmentation_transforms = configure_augmentations(config, [], resize_param)  # Create augmentation list without extending train_transforms

        # Apply the augmentations with the specified probability
        if config.augmentation.probs < 1.0:
            train_transforms.append(transforms.RandomApply([transforms.Compose(augmentation_transforms)],
                                                           p=config.augmentation.probs))
        else:
            train_transforms.extend(augmentation_transforms)

        train_transforms.append(transforms.ToTensor())  # Convert to tensor before applying tensor-based transformations

        # Add Random Erasing here after the ToTensor transformation
        if bool(config.augmentation.on) and bool(config.augmentation.cutout):
            train_transforms.append(MultipleRandomErasing(num_patches=config.augmentation.cutout_patch,
                                                          p=config.augmentation.cutout_probs,
                                                          scale=config.augmentation.cutout_scale,
                                                          ratio=config.augmentation.cutout_ratio,
                                                          value=config.augmentation.cutout_value))

        val_transforms.append(transforms.ToTensor())  # Validation transforms do not need the cutout

    else:
        # If augmentation is off, ensure basic transformations are still applied
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())

    # Compose all transforms into a single operation for train and validation
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    return train_transforms, val_transforms


class MultipleRandomErasing:
    def __init__(self, num_patches=3, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0):
        """
        Apply RandomErasing multiple times randomly.
        :param num_patches: Number of patches to erase.
        :param p: Probability of erasing a single patch.
        :param scale: Proportion of image to erase per patch.
        :param ratio: Aspect ratio of the erase region.
        :param value: Erasing value.
        """
        self.num_patches = num_patches
        self.eraser = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, img):
        for _ in range(self.num_patches):
            if random.uniform(0, 1) < self.eraser.p:
                img = self.eraser(img)
        return img


# Updated heavy_augmentation function to apply augmentations directly to an image
def heavy_augmentation(image, config):
    """
    Apply heavy augmentations to a single image using the given config.
    """
    resize_param = config.augmentation.resize_param

    # Define the augmentation pipeline
    augmentation_transforms = []
    augmentation_transforms.append(transforms.Resize((resize_param, resize_param)))

    # Add other augmentations based on the config
    augmentation_transforms = configure_heavy_augmentations(config, augmentation_transforms, resize_param)

    # Convert image to tensor
    augmentation_transforms.append(transforms.ToTensor())

    # Compose all transforms into a single operation
    augment_pipeline = transforms.Compose(augmentation_transforms)

    # Apply the transformations to the input image
    augmented_image = augment_pipeline(image)

    return augmented_image

# Configure heavy augmentations as before
def configure_heavy_augmentations(config, augmentation_transforms, resize_param):
    """
    Configure and return a list of heavy augmentation transforms.
    """
    augmentation_transforms.append(transforms.RandomApply(
        [transforms.RandomResizedCrop(size=(resize_param, resize_param), scale=(0.4, 0.7))],
        p=1.0))

    augmentation_transforms.append(transforms.RandomHorizontalFlip(p=config.augmentation.hf_probs))
    augmentation_transforms.append(transforms.RandomVerticalFlip(p=config.augmentation.vf_probs))

    augmentation_transforms.append(transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4)],
        p=config.augmentation.jitter_probs))

    augmentation_transforms.append(transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=1.0))
    augmentation_transforms.append(transforms.RandomApply([transforms.GaussianBlur(5, sigma=2.0)], p=1.0))

    return augmentation_transforms