import time as timing


class TimeSource:
    @property
    def seconds(self):
        return timing.time()

    @property
    def millis(self):
        return int(self.seconds * 1000)

    @property
    def nanos(self):
        return timing.time_ns()
