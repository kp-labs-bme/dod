from types import SimpleNamespace
from dataset.load_cifar10 import load_cifar10
from dataset.load_cifar100 import load_cifar100


def load_cifar(config: SimpleNamespace, data_dir: str, num_classes: int = 100) -> dict:
    if num_classes == 100:
        cifar_data = load_cifar100(config, data_dir)
    elif num_classes == 10:
        cifar_data = load_cifar10(config, data_dir)
    else:
        raise ValueError

    return cifar_data
