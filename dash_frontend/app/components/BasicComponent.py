import abc


class BasicComponent(object, metaclass=abc.ABCMeta):
    """
    All Dash components should have a layout
    """

    @abc.abstractmethod
    def layout(self):
        raise NotImplementedError()
