import torch
from typing import Tuple, Union
from torch.utils.data import DataLoader

from factory.base_factory import BaseFactory
from dataset.custom_sampler import CustomRandomSampler
from dataset.custom_data_loader import CustomDataLoader


class DataLoaderFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs) -> Tuple[DataLoader, DataLoader, Union[None, CustomRandomSampler]]:
        train_dataset, val_dataset = kwargs["train_dataset"], kwargs["val_dataset"]
        config = kwargs["config"]
        sorted_indexes = kwargs["sorted_indexes"]
        sorted_acces_path = kwargs["sorted_acces_path"]
        root_folder = kwargs["root_folder"]

        cuda_available = torch.cuda.is_available()
        device = 'cuda' if cuda_available else ''

        #     sampler = CustomRandomSampler(c_const=c_const,
        #                                     explore_type=explore_type,
        #                                     exploit_type=exploit_type,
        #                                     data_source=train_dataset)
        #     shuffle = False
        # else:
        #     sampler = None
        #     shuffle = True

        # train_loader = DataLoader(dataset=train_dataset,
        #                           batch_size=config.data_loader.batch_size_train,
        #                           num_workers=config.data_loader.num_workers,
        #                           sampler=sampler,
        #                           shuffle=shuffle,
        #                           pin_memory=cuda_available,
        #                           pin_memory_device=device,
        #                           )

        train_loader = CustomDataLoader(config,
                                        sorted_indexes=sorted_indexes,
                                        sorted_acces_path=sorted_acces_path,
                                        dataset=train_dataset,
                                        batch_size=config.data_loader.batch_size_train,
                                        num_workers=config.data_loader.num_workers,
                                        shuffle=True,
                                        pin_memory=cuda_available,
                                        pin_memory_device=device,
                                        dataset_path=root_folder)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=config.data_loader.batch_size_train,
                                num_workers=config.data_loader.num_workers,
                                pin_memory=cuda_available,
                                pin_memory_device=device
                                )

        return train_loader, val_loader #, sampler
