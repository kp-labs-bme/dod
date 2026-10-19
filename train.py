import os
import torch

from factory.dataset_factory import DatasetFactory
from factory.data_loader_factory import DataLoaderFactory
from factory.model_factory import ModelFactory
from factory.optim_factory import AgentFactory
from factory.callbacks_factory import CallbacksFactory
from trainer.trainer import Trainer
from utils.custom_argument_parser import CustomArgParser


def main():
    path = os.path.abspath('configuration.ini')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n\nUsing {device}\n\n")

    arg_parser = CustomArgParser(config_path=path)
    parsed_config = arg_parser.get_parsed_config()

    if parsed_config.data_loader.use_selection:
      with open(parsed_config.data_loader.selection_name + "_exploit_indexes.txt", "r") as f:
        lines = f.readlines()

      # Parse the lines to extract the indexes and exploit values
      parsed_data = [line.strip().split(",") for line in lines]
      sorted_indexes = [int(item[0]) for item in parsed_data]
      sorted_acces_path = [item[2] for item in parsed_data]
    else:
       sorted_indexes = None
       sorted_acces_path = None

    Trainer.seed(parsed_config.trainer.seed)
    my_model = ModelFactory.create(config=parsed_config)
    my_callbacks = CallbacksFactory.create(config=parsed_config, model=my_model, config_path=path, device=device)

    train_dataset, val_dataset, root_folder = DatasetFactory.create(config=parsed_config)
    train_loader, val_loader = DataLoaderFactory.create(config=parsed_config, 
                                                        train_dataset=train_dataset,
                                                        val_dataset=val_dataset, 
                                                        sorted_indexes=sorted_indexes,
                                                        sorted_acces_path=sorted_acces_path,
                                                        root_folder=root_folder)
    
    my_criterion, my_optimizer, lr_scheduler = AgentFactory.create(config=parsed_config, model=my_model)

    trainer = Trainer(config=parsed_config,
                      criterion=my_criterion,
                      optimizer=my_optimizer,
                      lr_scheduler=lr_scheduler,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      callbacks=my_callbacks,
                      model=my_model,
                    #   sampler=my_sampler
                      )

    trainer.train(device=device)


if __name__ == "__main__":
    main()
