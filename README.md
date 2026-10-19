# DOD

A PyTorch project for identifying potentially problematic training samples during supervised image classification. It trains a classifier, tracks a score for each training image, and exports a ranked list of samples for inspection or later selection experiments.

The code supports prediction error, prediction changes, and uncertainty as scoring methods. It also provides experiments with incorrectly labelled duplicate images and heavily augmented images.

## Installation

From the project directory, create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install requests psutil
```

`requests` and `psutil` are imported by the code but are currently missing from `requirements.txt`. PyTorch and torchvision must be compatible with each other. Training uses CUDA when available and otherwise runs on the CPU.

## Running the code

Run commands from the project directory, where `train.py` and `configuration.ini` are located.

Start a new training run without Neptune logging:

```bash
python train.py \
  --data_loader_use_selection False \
  --callbacks_neptune_logger_callback False \
  --optim_lr_decay False
```

This uses the default CIFAR-100 configuration. Missing data is downloaded automatically, and CIFAR-100 is also saved as images in class folders. The default experiment adds 1,000 incorrectly labelled duplicates and 1,000 heavily augmented samples. To train without these additions, append `--data_loader_n_duplicate 0 --data_loader_n_undoable 0`.

For a short trial, append `--trainer_n_epochs 2`. Sample scores start updating after the first epoch with the default settings.

The command disables selection because the supplied configuration expects an existing `base_exploit_indexes.txt` file. Learning rate decay is disabled here because the trainer currently advances the scheduler only when Neptune logging is enabled.

### Configuration

Edit [configuration.ini](configuration.ini) to change the experiment settings. Individual settings can also be overridden using `--<section>_<key> <value>`:

| Setting | Purpose |
| --- | --- |
| `trainer.n_epochs` | Number of training epochs |
| `trainer.seed` | Random seed |
| `data_loader.data_folder_name` | Dataset; use `Cifar100` or `TinyImagenet` for the path-based training workflow |
| `data_loader.batch_size_train` | Training batch size |
| `data_loader.exploit_type` | Sample scoring: `pb_error`, `label_change`, or `uncertainty` |
| `data_loader.n_duplicate` / `n_undoable` | Number of incorrectly labelled duplicates / heavily augmented samples |
| `data_loader.selection_name` | Prefix for the exported ranking file |
| `augmentation.*` | Resizing, cropping, flips, colour jitter, blur, and cutout settings |
| `model.model_type` | Classifier architecture |
| `model.pretrained` | Load pretrained torchvision weights |
| `optim.*` | Loss, optimizer, learning rate, and scheduler settings |

Available models are `resnet50`, `mobilenet_v3_large`, `mobilenet_v3_small`, `efficientnet_b0`, and `efficientnet_b1`. Optimizers include SGD, Adam, and RMSprop; losses include cross entropy and focal loss.

For example, add `--model_model_type resnet50 --optim_learning_rate 0.001` to the training command. Run `python train.py --help` to list the available arguments.

CIFAR-10 and ImageNet loaders are also included, but their integration with the current training pipeline needs adjustment.

### Results

Training prints loss and accuracy for the training and validation sets. With the default custom scoring enabled and selection disabled, it writes `<selection_name>_exploit_indexes.txt` at the end of training (`base_exploit_indexes.txt` by default).

Each line contains `sample_index,score,image_path`, ordered from highest to lowest score. Training batches are shuffled; the scores are used for the exported ranking.

Neptune logging can be enabled in the `[callbacks]` section with your own project and API token. It records metrics and saves the model through the Neptune integration. With logging disabled, the code does not save a local model checkpoint. The logger uploads the configuration and its fields, so review credential handling before enabling it.

## Code structure

```text
dod-main/
├── train.py                 # Training entry point
├── configuration.ini        # Experiment settings
├── requirements.txt         # Python dependencies
├── factory/                 # Builds datasets, loaders, models, optimizers, and callbacks
├── trainer/
│   └── trainer.py           # Training, validation, and ranking export
├── dataset/                 # Dataset loading, wrappers, and sample scoring
│   └── custom_data_loader.py
├── data/
│   └── download/            # Dataset download helpers; datasets are stored under data/
├── callbacks/
│   └── neptune_logger.py    # Neptune experiment logging
├── utils/
│   ├── augmentation.py      # Image augmentation pipelines
│   ├── custom_argument_parser.py
│   └── optim/               # Focal loss and learning rate scheduler
└── static/                  # Images, styles, fonts, and JavaScript assets
```

## License

This project is released under the [MIT License](LICENSE).
