import io
import sys
import types
import argparse
from typing import Union
import neptune.new as neptune
from torchsummary import summary
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger:
    def __init__(self, **kwargs):
        token = kwargs["token"]
        project = kwargs["project"]
        self.config = kwargs["config"]
        self.config_path = kwargs["config_path"]
        self.device = kwargs["device"]
        self.run = neptune.init_run(api_token=token, project=project)
        self.logger = None

    def log_config_to_neptune(self) -> None:
        with open(self.config_path, 'r') as file:
            self.run["config_file"].upload(self.config_path)

        self.log_nested_parameters("config", self.config)

    def log_nested_parameters(self, key_path, namespace) -> None:
        if isinstance(namespace, types.SimpleNamespace):
            for key, value in vars(namespace).items():
                new_key_path = f"{key_path}/{key}"
                if isinstance(value, types.SimpleNamespace):
                    # Recursive call for nested Namespace
                    self.log_nested_parameters(new_key_path, value)
                else:
                    # Convert to supported type and log
                    self.run[new_key_path] = self.convert_to_supported_type(value)
        elif isinstance(namespace, dict):
            for key, value in namespace.items():
                new_key_path = f"{key_path}/{key}"
                if isinstance(value, (types.SimpleNamespace, dict)):
                    # Recursive call for nested Namespace or dict
                    self.log_nested_parameters(new_key_path, value)
                else:
                    # Convert to supported type and log
                    self.run[new_key_path] = self.convert_to_supported_type(value)
        else:
            # If not a namespace or dict, simply log the value directly
            self.run[key_path] = self.convert_to_supported_type(namespace)

    def convert_to_supported_type(self, value) -> Union[list, dict, str]:
        if isinstance(value, argparse.Namespace):
            return vars(value)  # Convert Namespace to dict
        if isinstance(value, list) or isinstance(value, tuple):
            return str(value)  # Convert list or tuple to string
        if isinstance(value, dict):
            return {k: self.convert_to_supported_type(v) for k, v in value.items()}  # Convert each item in the dict
        return value

    def log_model_summary(self, model, input_size) -> None:
        # Redirect stdout to capture the output of the summary
        summary_str_io = io.StringIO()  # Create a StringIO object to capture output
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = summary_str_io  # Redirect standard output to the StringIO object

        try:
            summary(model.to(self.device), input_size)  # Generate summary without verbose argument
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value

        summary_str = summary_str_io.getvalue()  # Get the string from StringIO
        summary_file = 'model_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_str)  # Write the summary to a text file

        # Upload the model summary to Neptune
        self.run["model/model_summary"].upload(summary_file)

        summary_str_io.close()  # Close StringIO object

    def start_logging(self, model) -> None:
        dim = self.config.augmentation.resize_param
        input_size = (3, dim, dim)
        self.log_config_to_neptune()  # Log the config before starting
        self.log_model_summary(model, input_size)  # Log model summary
        self.logger = NeptuneLogger(run=self.run, model=model)

    def save_checkpoint(self, checkpoint_name) -> None:
        self.logger.save_checkpoint(checkpoint_name=checkpoint_name)

    def save_model(self) -> None:
        self.logger.save_model()

    def stop(self) -> None:
        self.run.stop()
