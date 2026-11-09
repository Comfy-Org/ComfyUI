from typing import TypeVar

class SingletonMetaclass(type):
    T = TypeVar("T", bound="SingletonMetaclass")
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaclass, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: T) -> None:
        assert cls not in SingletonMetaclass._instances, (
            "Cannot inject instance after first instantiation"
        )
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args, **kwargs) -> T:
        """
        Gets the singleton instance of the class, creating it if it doesn't exist.
        """
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super(
                SingletonMetaclass, cls
            ).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProxiedSingleton(object, metaclass=SingletonMetaclass):
    def __init__(self):
        super().__init__()
