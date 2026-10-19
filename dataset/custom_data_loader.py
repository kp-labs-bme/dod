import os
import torch
from torch.utils.data import DataLoader
import numpy as np

class CustomDataLoader(DataLoader):
    def __init__(self, config, sorted_indexes=None, sorted_acces_path=None, dataset_path=None, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.remove_count = config.data_loader.remove_count
        self.use_selection = config.data_loader.use_selection
        self.current_epoch = 0

        self.sorted_indexes = sorted_indexes
        self.sorted_acces_path = sorted_acces_path

        # Set dataset root directory path
        print(dataset_path)
        print(os.getcwd())
        self.dataset_root_dir = os.path.join(os.getcwd(), os.path.dirname(dataset_path))

        num_samples = len(self.dataset)
        self.original_indices = np.arange(num_samples)  # Store original indices
        self.fit_counts = torch.zeros(num_samples, dtype=torch.int32, device=self.device)
        self.init_pb_error = 0.001
        self.indexes = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        self.cp = config.data_loader.c_constant
        self.labels = -1 * torch.ones(num_samples, dtype=torch.double, device=self.device)
        self.explore_type = config.data_loader.explore_type
        self.exploit_type = config.data_loader.exploit_type
        self.pb_discount = config.data_loader.pb_discount
        self.pb_log = {}

        # Initialize exploit based on type
        if self.exploit_type == "label_change":
            self.exploit = torch.zeros(num_samples, dtype=torch.double, device=self.device)
        elif self.exploit_type == "pb_error":
            self.exploit = torch.ones(num_samples, dtype=torch.double, device=self.device) * self.init_pb_error
        elif self.exploit_type == "uncertainty":
            self.exploit = torch.ones(num_samples, dtype=torch.double, device=self.device) * 0.5
        else:
            raise ValueError(f"Exploit type '{self.exploit_type}' invalid. Can be 'pb_error', 'label_change', or 'uncertainty'")

        # Ensure to remove the originals of the augmented images in the next run when use_selection is True
        if self.use_selection:
            print(f"\nRemoving heavily augmented originals and first {self.remove_count} indexes ...\n")
            self._remove_augmented_originals()
            self._remove_top_n_samples()

    def _remove_augmented_originals(self):
        """Remove the original images that were heavily augmented in the previous run."""
        
        # Correct path to the 'undoable' directory inside the dataset root
        undoable_dir = os.path.join(self.dataset_root_dir, 'undoable')
        if not os.path.exists(undoable_dir):
            print(f"No undoable directory found at {undoable_dir}, skipping original removal.")
            return

        undoable_paths = []
        # Collect all paths of augmented images
        for class_folder in os.listdir(undoable_dir):
            class_folder_path = os.path.join(undoable_dir, class_folder)
            for image_file in os.listdir(class_folder_path):
                undoable_paths.append(os.path.join(class_folder_path, image_file))

        # Adjust dataset paths
        dataset_paths = [os.path.join(self.dataset_root_dir, os.path.relpath(path)) for path in self.dataset.data]

        # Create a mask to keep track of which original samples to remove
        mask = np.ones(len(self.dataset), dtype=bool)
        
        # Mark the originals for removal
        for augmented_path in undoable_paths:
            for i, dataset_path in enumerate(dataset_paths):
                # Compare the filenames (removing the 'undoable_' prefix)
                if os.path.basename(augmented_path).replace("undoable_", "") == os.path.basename(dataset_path):
                    mask[i] = False  # Mark for removal

        # Apply mask to remove original samples
        self.dataset.data = [self.dataset.data[i] for i in range(len(self.dataset.data)) if mask[i]]
        self.dataset.labels = [self.dataset.labels[i] for i in range(len(self.dataset.labels)) if mask[i]]

        # Update attributes after removal
        self.original_indices = np.arange(len(self.dataset))
        self.fit_counts = torch.zeros(len(self.dataset), dtype=torch.int32, device=self.device)
        self.labels = -1 * torch.ones(len(self.dataset), dtype=torch.double, device=self.device)
        
        if self.exploit_type == "label_change":
            self.exploit = torch.zeros(len(self.dataset), dtype=torch.double, device=self.device)
        else:
            self.exploit = torch.ones(len(self.dataset), dtype=torch.double, device=self.device) * self.init_pb_error

    def _remove_top_n_samples(self):
        """Remove the first `remove_count` samples based on the sorted access paths."""
        
        if self.remove_count > len(self.sorted_acces_path):
            raise ValueError("remove_count is greater than the number of available sorted access paths.")
        
        # Use the correct project root (TinyImagenet directory)
        full_sorted_paths = [os.path.join(self.dataset_root_dir, path) for path in self.sorted_acces_path[:self.remove_count]]
        
        # Adjust dataset paths to match the dataset directory structure
        dataset_paths = [os.path.join(self.dataset_root_dir, os.path.relpath(path)) for path in self.dataset.data]

        # Create a mask to mark the samples to remove
        mask = np.ones(len(self.dataset), dtype=bool)
        
        # Remove samples based on matching access paths
        for sorted_path in full_sorted_paths:
            for i, dataset_path in enumerate(dataset_paths):
                if sorted_path == dataset_path:
                    mask[i] = False  # Mark for removal
                    break  # Exit the loop once a match is found
        
        # Apply mask to remove samples
        self.dataset.data = [self.dataset.data[i] for i in range(len(self.dataset.data)) if mask[i]]
        self.dataset.labels = [self.dataset.labels[i] for i in range(len(self.dataset.labels)) if mask[i]]
        
        # Update attributes after removal
        self.original_indices = np.arange(len(self.dataset))
        self.fit_counts = torch.zeros(len(self.dataset), dtype=torch.int32, device=self.device)
        self.labels = -1 * torch.ones(len(self.dataset), dtype=torch.double, device=self.device)
        
        if self.exploit_type == "label_change":
            self.exploit = torch.zeros(len(self.dataset), dtype=torch.double, device=self.device)
        else:
            self.exploit = torch.ones(len(self.dataset), dtype=torch.double, device=self.device) * self.init_pb_error

    def _check_access_path_matches(self):
        """Check if the access path-index pairs from sorted indexes and sorted access paths match with the current dataset."""
        
        if self.sorted_indexes is None or self.sorted_acces_path is None:
            print("No sorted indexes or sorted access paths provided.")
            return
        
        # Extract only the project-root relative part from dataset.data and sorted_acces_path
        dataset_access_paths = [os.path.relpath(path, start=self.sorted_acces_path[0]) for path in self.dataset.data]
        
        # Ensure sorted_acces_path and sorted_indexes are of the same length
        if len(self.sorted_acces_path) != len(self.sorted_indexes):
            raise ValueError("Mismatch in length between sorted access paths and sorted indexes.")
        
        # Counters for matches and mismatches
        match_count = 0
        mismatch_count = 0
        
        # Compare index-access path pairs
        for idx, sorted_index in enumerate(self.sorted_indexes):
            sorted_access_path = self.sorted_acces_path[idx]  # Access path at sorted index
            
            # Get the corresponding access path from the current dataset using the sorted index
            dataset_access_path = dataset_access_paths[sorted_index]
            
            # Check if they match
            if sorted_access_path == dataset_access_path:
                match_count += 1
            else:
                mismatch_count += 1
                print(f"Mismatch at index {sorted_index}: sorted access path = {sorted_access_path}, dataset access path = {dataset_access_path}")
        
        # Log the results
        print(f"\nTotal matches: {match_count}")
        print(f"Total mismatches: {mismatch_count}")

    def update_metrics(self, indexes: torch.tensor, outputs: torch.tensor, predicted: torch.tensor, labels) -> dict:
        # Ensure the tensors are on the correct device
        indexes = indexes.to(self.device)
        labels = labels.to(self.device)
        predicted = predicted.to(self.device)

        # Detach outputs from the graph (if no backpropagation is needed)
        outputs = outputs.detach()

        if self.explore_type == "fit_count":
            self.fit_counts[indexes] += 1
            log_term = torch.log(torch.max(self.fit_counts) + self.init_pb_error)
            explore = self.cp * torch.sqrt(2 * log_term / (self.fit_counts[indexes] + self.init_pb_error))
        else:
            raise NotImplementedError("Explore component not implemented")

        if self.exploit_type == "label_change":
            indexes_numpy = indexes.detach().cpu().numpy()
            labels_dict = dict(zip(indexes_numpy, predicted.cpu().numpy()))  # Detach predicted

            for index in indexes_numpy:
                if labels_dict[index] != self.labels[index].item():
                    self.exploit[index] += 1
                    self.labels[index] = labels_dict[index]
                elif labels_dict[index] == self.labels[index].item() and self.exploit[index] == 0:
                    self.exploit[index] += 1

            exploit = self.exploit[indexes_numpy] / torch.max(self.exploit)

        elif self.exploit_type == "pb_error":
            with torch.no_grad():
                outputs = torch.nn.functional.softmax(outputs, dim=1).to(self.device)
                mask = (labels == predicted).float()

                pb_error = ((1 - mask) * outputs[torch.arange(outputs.size(0)), predicted] +
                            mask * (1 - outputs[torch.arange(outputs.size(0)), predicted]))

                # Ensure pb_error and exploit are detached from any computational graph
                pb_error = pb_error.detach()

                # Update exploit with detached pb_error
                self.exploit[indexes] = self.pb_discount * self.exploit[indexes] + (1 - self.pb_discount) * pb_error.double()

                exploit = self.exploit[indexes]

        elif self.exploit_type == "uncertainty":
            outputs = torch.nn.functional.softmax(outputs, dim=1).to(self.device).detach()  # Ensure outputs are detached
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)
            top2_probs, top2_indices = torch.topk(outputs, 2, dim=1)
            true_probs = outputs[torch.arange(outputs.size(0)), labels]
            penalty = (top2_probs[:, 0] - true_probs) * (labels != top2_indices[:, 0])

            uncertainty_metric = 0.5 * entropy + 0.5 * penalty

            # Detach uncertainty_metric from the graph
            uncertainty_metric = uncertainty_metric.detach()

            # Bellman-like update for uncertainty
            self.exploit[indexes] = self.pb_discount * self.exploit[indexes] + (1 - self.pb_discount) * uncertainty_metric.double()
            exploit = self.exploit[indexes]

        else:
            raise NotImplementedError("Exploit component not implemented")

        with torch.no_grad():
            # Extract scalar values using .item() to prevent tensor-related memory issues
            self.pb_log["exploit_mean"] = exploit.mean().item()
            self.pb_log["exploit_std"] = exploit.std().item()
            self.pb_log["exploit_max"] = exploit.max().item()
            self.pb_log["exploit_min"] = exploit.min().item()

        return self.pb_log

    def get_exploit_values(self):
        return self.exploit.detach().cpu().numpy()  # Detach and move to CPU

