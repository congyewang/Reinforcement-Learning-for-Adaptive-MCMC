from stable_baselines3.common.type_aliases import Schedule
import math


class LearningRateSchedule:
    @staticmethod
    def linear_schedule(initial_lr: float = 0.001) -> Schedule:
        """Linear learning rate schedule.

        Args:
            initial_lr (torch.Tensor): Initial learning rate.

        Returns (Callable[[float], float]) aka Schedule in sb3):
            schedule that computes current learning rate depending on remaining progress.
        """

        def schedule(progress_remaining: float) -> float:
            """Progress will decrease from 1 (beginning) to 0.

            Args:
                Progress_remaining (float)
            Returns (float):
                Current learning rate depending on remaining progress.
            """
            return progress_remaining * initial_lr

        return schedule

    @staticmethod
    def piecewise_linear_schedule(
        initial_lr: float = 0.001, transition_point=0.5, end_lr=0.0
    ):
        """
        A scheduling function in which the learning rate decreases rapidly in the first half of training and slowly in the second half.

        Args:
            initial_lr: Initial learning rate
            transition_point: Turning points for fast descents and slow descents (as a proportion of total training steps)
            end_lr: Learning rate at the end of training

        Returns:
            Callable learning rate schedule
        """

        def schedule(progress_remaining: float) -> float:
            if progress_remaining > transition_point:
                # Linear decline before the turning point
                return initial_lr - (initial_lr - end_lr) * (
                    (1 - progress_remaining) / transition_point
                )
            else:
                # Slow decline after the turning point
                return end_lr

        return schedule

    @staticmethod
    def slowness_speedy_learning_rate_schedule(
        initial_lr, transition_point=0.5, end_lr=0.0
    ):
        """
        A scheduling function in which the learning rate decreases slowly in the first half of training and rapidly in the second half

        Args:
            initial_lr: Initial learning rate
            transition_point: Turning points for slow descents and fast descents (as a proportion of total training steps)
            end_lr: Learning rate at the end of training

        Return:
            Callable learning rate schedule
        """

        def schedule(progress_remaining: float) -> float:
            if progress_remaining > transition_point:
                # Slow decline before turning point
                return (
                    initial_lr
                    - (initial_lr - end_lr)
                    * ((1 - progress_remaining) / transition_point) ** 2
                )
            else:
                # Rapid decline after the turning point
                return (
                    end_lr
                    + (initial_lr - end_lr)
                    * (progress_remaining / (1 - transition_point)) ** 2
                )

        return schedule

    @staticmethod
    def exponential_decay_schedule(
        initial_lr: float = 0.001, decay_rate: float = 0.96, decay_steps: int = 10000
    ):
        """
        Exponential decay learning rate scheduling

        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate
            decay_steps: Decay Steps

        Return:
            Callable learning rate schedule
        """

        def schedule(progress_remaining: float) -> float:
            return initial_lr * (decay_rate ** ((1 - progress_remaining) * decay_steps))

        return schedule

    @staticmethod
    def cosine_annealing_schedule(
        initial_lr: float = 0.001, min_lr: float = 1e-4, total_timesteps: float = 10_000
    ):
        """
        Cosine annealing learning rate scheduling

        Args:
            initial_lr: Initial learning rate.
            min_lr: Minimum learning rate
            total_timesteps: Total training steps.

        Return:
            Callable learning rate schedule
        """

        def schedule(progress_remaining: float) -> float:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
            decayed = (1 - min_lr / initial_lr) * cosine_decay + min_lr / initial_lr
            return initial_lr * decayed

        return schedule
