from abc import ABC, abstractmethod
from rlmcmc.env import RLMHEnvV31, RLMHEnvV31A, RLMHEnvV31B


class TestRLMHEnvBase(ABC):
    @abstractmethod
    def test_log_target_pdf(self):
        pass

    @abstractmethod
    def test_init(self):
        pass

    @abstractmethod
    def test_distance_function(self):
        pass

    @abstractmethod
    def reward_function(self):
        pass

    @abstractmethod
    def test_step(self):
        pass


class TestRLMHEnvV31(TestRLMHEnvBase):
    def test_log_target_pdf(self):
        pass

    def test_init(self):
        pass

    def test_distance_function(self):
        pass

    def reward_function(self):
        pass

    def test_step(self):
        pass
