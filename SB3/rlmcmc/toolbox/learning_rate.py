from stable_baselines3.common.type_aliases import Schedule


def linear_schedule(initial_value: float = 0.001) -> Schedule:
    """Linear learning rate schedule.

    Args:
        initial_value (torch.Tensor): Initial learning rate.

    Returns (Callable[[float], float]) aka Schedule in sb3):
        schedule that computes current learning rate depending on remaining progress.
    """

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0.

        Args:
            Progress_remaining (float)
        Returns (float):
            Current learning rate depending on remaining progress.
        """
        return progress_remaining * initial_value

    return func
