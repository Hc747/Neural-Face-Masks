import time as timing


class TimeSource:
    @property
    def seconds(self) -> float:
        return timing.time()

    @property
    def millis(self) -> int:
        return int(self.seconds * 1000)

    @property
    def nanos(self) -> int:
        return timing.time_ns()
