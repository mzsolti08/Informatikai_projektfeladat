import time

class ActivityStats:

    def __init__(self):

        self.stats = {}

        self.last_activity = None
        self.last_time = time.time()

    def update(self, activity):

        now = time.time()

        if self.last_activity is not None:

            elapsed = now - self.last_time

            if self.last_activity not in self.stats:
                self.stats[self.last_activity] = 0

            self.stats[self.last_activity] += elapsed

        self.last_activity = activity
        self.last_time = now

    def get_stats(self):

        result = []

        for activity, seconds in self.stats.items():

            total = int(seconds)

            minutes = total // 60
            secs = total % 60

            result.append(
                f"{activity}: {minutes:02}:{secs:02}"
            )

        return result