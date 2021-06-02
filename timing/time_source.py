import time as timing

"""
A module exporting utility functions for measuring periods of time.
"""


class TimeSource:
    """
    An object that allows for the measuring of time at varying levels of precision (seconds, milliseconds, nanoseconds).
    """

    @property
    def seconds(self) -> float:
        """
        The current timestamp in seconds since the unix epoch
        """
        return timing.time()

    @property
    def millis(self) -> int:
        """
        The current timestamp in milliseconds since the unix epoch
        """
        return int(self.seconds * 1000)

    @property
    def nanos(self) -> int:
        """
        The current timestamp in nanoseconds since the unix epoch
        """
        return timing.time_ns()
