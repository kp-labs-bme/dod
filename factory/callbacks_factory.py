from factory.base_factory import BaseFactory
from callbacks.neptune_logger import CustomNeptuneLogger


class CallbacksFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs) -> dict:
        my_callbacks = {}
        config = kwargs["config"]
        config_path = kwargs["config_path"]
        device = kwargs["device"]

        if config.callbacks.neptune_logger_callback:
            neptune_logger_callback = CustomNeptuneLogger(token=config.callbacks.neptune_token,
                                                          project=config.callbacks.neptune_project,
                                                          config=config,
                                                          config_path=config_path,
                                                          device=device)
            neptune_logger_callback.start_logging(kwargs["model"])
            my_callbacks["neptune_logger"] = neptune_logger_callback

        return my_callbacks
