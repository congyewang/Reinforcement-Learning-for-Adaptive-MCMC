from abc import ABCMeta, abstractmethod


class QFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, state_dim, action_dim):
        pass

    @abstractmethod
    def __call__(self, state, action):
        pass

    @abstractmethod
    def grad(self, state_t, action):
        pass

class PolicyFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, state_dim, action_dim):
        pass

    @abstractmethod
    def __call__(self, state, action):
        pass

    @abstractmethod
    def grad(self, state_t, action):
        pass
