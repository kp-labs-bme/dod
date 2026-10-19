import configparser
from argparse import ArgumentParser
from types import SimpleNamespace
import ast


class CustomArgParser:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.arg_parser = ArgumentParser()
        self.config_namespace = SimpleNamespace()

    def add_dynamic_args(self):
        for section in self.config.sections():
            if not hasattr(self.config_namespace, section):
                setattr(self.config_namespace, section, SimpleNamespace())
            section_namespace = getattr(self.config_namespace, section)
            for key, value in self.config.items(section):
                # Try to automatically determine the type of the argument
                try:
                    # Attempt to interpret the value in a Python literal sense
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # Fall back to string if interpretation fails
                    pass

                # Check if the value is boolean, and handle it appropriately
                if isinstance(value, bool):
                    self.arg_parser.add_argument(f"--{section}_{key}", type=lambda x: x.lower() in ['true', '1'], default=value,
                                                help=f"{section}: {key} (default: {value})")
                else:
                    self.arg_parser.add_argument(f"--{section}_{key}", type=type(value), default=value,
                                             help=f"{section}: {key} (default: {value})")
                # Add the same configuration to the namespace
                setattr(section_namespace, key, value)

    def get_parsed_config(self):
        self.add_dynamic_args()
        
        # Parse the known arguments and ignore unrecognized ones
        parsed_args, _ = self.arg_parser.parse_known_args()

        # Update namespace with parsed arguments
        for section in self.config.sections():
            section_namespace = getattr(self.config_namespace, section)
            for key in self.config.options(section):
                arg_key = f"{section}_{key}"
                if hasattr(parsed_args, arg_key):
                    value = getattr(parsed_args, arg_key)
                    setattr(section_namespace, key, value)

        return self.config_namespace
