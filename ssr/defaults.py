from typing import Callable


class DefaultValue:
    pass


DEFAULT_VALUE = DefaultValue()


def underload(func: Callable, default_attr: str = "defaults"):
    def wrapper(self, *args, **kwargs):
        defaults = getattr(self, default_attr).model_dump()
        new_values = {}

        for key, _ in func.__annotations__.items():
            if key in defaults:
                new_values[key] = kwargs.get(key, defaults[key])
            elif key in kwargs:
                new_values[key] = kwargs[key]

        for value, (key, _) in zip(args, func.__annotations__.items()):
            new_values[key] = value

        return getattr(self, func.__name__ + "_")(**new_values)

    return wrapper
