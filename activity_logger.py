from datetime import datetime

class ActivityLogger:
    def __init__(self, filename="activity_log.txt"):
        self.filename = filename
        self.last_entry = ""

    def log(self, person, activity):

        entry = f"{person} - {activity}"

        if entry != self.last_entry:

            now = datetime.now().strftime("%H:%M:%S")

            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{now} - {person} - {activity}\n")

            self.last_entry = entry