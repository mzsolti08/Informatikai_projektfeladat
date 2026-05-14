from datetime import datetime

class ActivityLogger:
    def __init__(self, filename="activity_log.txt"):
        self.filename = filename
        self.last_activity = ""

    def log(self, activity):
        if activity != self.last_activity:

            now = datetime.now().strftime("%H:%M:%S")

            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{now} - {activity}\n")

            self.last_activity = activity