import numpy as np
import random
import torch
import pickle
import copy
import os
import psutil


class Trainer(object):

    def __init__(self, config, criterion, optimizer, lr_scheduler, train_loader, val_loader, callbacks, model):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.model = model

    @staticmethod
    def seed(seed_value: int):
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)

    def train(self, device):
        n_epochs = self.config.trainer.n_epochs

        if self.config.callbacks.neptune_logger_callback:
            neptune = self.callbacks["neptune_logger"]
            num_training_images = len(self.train_loader.dataset)
            neptune.run[neptune.logger.base_namespace]["parameters/num_training_images"].log(num_training_images)

        # Training & Validation
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []

        for epoch in range(n_epochs):
            self.model.train()
            running_loss, correct_train = 0.0, 0

            for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
                inputs, labels, indices = inputs.to(device), labels.to(device), indices.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)  # Removed .clone() here

                # Neptune logging (log only at intervals)
                if self.config.callbacks.neptune_logger_callback and batch_idx % self.config.trainer.log_interval == 0:
                    neptune.run[neptune.logger.base_namespace]["metrics/train_loss"].append(loss.item())

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()

                if batch_idx % self.config.trainer.log_interval == 0:
                    print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * len(inputs), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))

                if self.config.data_loader.custom_loader and epoch > self.config.data_loader.prio_patience:
                    self.train_loader.update_metrics(indices.cpu(), outputs.cpu(), predicted.cpu(), labels.cpu())
                    
                # Log memory after each batch
                if self.config.callbacks.neptune_logger_callback:
                    memory = psutil.virtual_memory()
                    neptune.run[neptune.logger.base_namespace]["memory/total_memory"].append(memory.total)
                    neptune.run[neptune.logger.base_namespace]["memory/available_memory"].append(memory.available)
                    neptune.run[neptune.logger.base_namespace]["memory/used_memory"].append(memory.used)
                    neptune.run[neptune.logger.base_namespace]["memory/free_memory"].append(memory.free)
                    neptune.run[neptune.logger.base_namespace]["memory/percent_memory"].append(memory.percent)

                # Move predicted tensor to CPU after using it to free GPU memory
                predicted = predicted.detach().cpu()

                # Log memory after each batch (optional)
                if self.config.callbacks.neptune_logger_callback and batch_idx % self.config.trainer.log_interval == 0:
                    memory = psutil.virtual_memory()
                    neptune.run[neptune.logger.base_namespace]["memory/total_memory"].append(memory.total)
                    neptune.run[neptune.logger.base_namespace]["memory/available_memory"].append(memory.available)
                    neptune.run[neptune.logger.base_namespace]["memory/used_memory"].append(memory.used)
                    neptune.run[neptune.logger.base_namespace]["memory/free_memory"].append(memory.free)
                    neptune.run[neptune.logger.base_namespace]["memory/percent_memory"].append(memory.percent)

            train_loss.append(running_loss / len(self.train_loader))
            last_train_acc = 100. * correct_train / len(self.train_loader.dataset)
            train_accuracy.append(last_train_acc)

            if self.config.callbacks.neptune_logger_callback:
                neptune.run[neptune.logger.base_namespace]["metrics/train_acc"].append(last_train_acc / 100.)

            self.model.eval()
            running_loss, correct_val = 0.0, 0

            with torch.no_grad():
                for inputs, labels, _ in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if self.config.callbacks.neptune_logger_callback:
                        neptune.run[neptune.logger.base_namespace]["metrics/val_loss"].append(loss.item())

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_val += (predicted == labels).sum().item()

                    # Move predicted and labels tensors to CPU after using them to free GPU memory
                    predicted = predicted.cpu()
                    labels = labels.cpu()

            val_loss.append(running_loss / len(self.val_loader))
            last_val_acc = 100. * correct_val / len(self.val_loader.dataset)
            val_accuracy.append(last_val_acc)

            if self.config.callbacks.neptune_logger_callback:
                neptune.run[neptune.logger.base_namespace]["metrics/val_acc"].append(last_val_acc / 100.)

            print(
                f"Epoch {epoch + 1}/{n_epochs} - Train loss: {train_loss[-1]:.4f},"
                f"Train accuracy: {train_accuracy[-1]:.2f}%, Val loss: {val_loss[-1]:.4f},"
                f"Val accuracy: {val_accuracy[-1]:.2f}%")

            if self.config.optim.lr_decay and self.config.callbacks.neptune_logger_callback:
                neptune.run[neptune.logger.base_namespace]["metrics/lr"].append(self.lr_scheduler.get_last_lr()[0])
                self.lr_scheduler.step()

            # Free unused GPU memory after each epoch
            torch.cuda.empty_cache()

        if self.config.callbacks.neptune_logger_callback:
            self.callbacks["neptune_logger"].save_model()

        if self.config.data_loader.custom_loader and not self.config.data_loader.use_selection:
            exploit_values = self.train_loader.get_exploit_values()
            exploit_values_list = exploit_values.tolist()
            dataset_indexes = np.arange(len(exploit_values))
            sorted_indexes = dataset_indexes[np.argsort(-exploit_values)]
            sorted_indexes_list = sorted_indexes.tolist()
            sorted_exploit_values_list = exploit_values[sorted_indexes].tolist()

            image_paths = [self.train_loader.dataset.data[idx] for idx in sorted_indexes_list]

            # Strip the path before 'data/' to keep only the relevant part
            image_paths_cleaned = [img_path[img_path.index("data/"):] for img_path in image_paths]

            # Create the lines with index, exploit value, and cleaned image path
            lines = [f"{index},{value},{img_path}" for index, value, img_path in zip(sorted_indexes_list, sorted_exploit_values_list, image_paths_cleaned)]
            text_file_name = self.config.data_loader.selection_name + "_exploit_indexes.txt"

            with open(text_file_name, "w") as f:
                f.write("\n".join(lines))

            if self.config.callbacks.neptune_logger_callback:
                self.callbacks["neptune_logger"].run["exploit_values"].log(exploit_values_list)
                self.callbacks["neptune_logger"].run["files/selection"].upload(text_file_name)
                self.callbacks["neptune_logger"].stop()
