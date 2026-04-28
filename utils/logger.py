import datetime

LOG_FILE = "data/logs.txt"

def log_event(name, activity):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now} - {name} - {activity}\n")