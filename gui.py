import tkinter as tk
from threading import Thread
from main import main

def start_system():
    Thread(target=main).start()

def stop_app():
    root.destroy()

root = tk.Tk()
root.title("Face Recognition System")
root.geometry("300x200")

btn_start = tk.Button(root, text="Start", command=start_system)
btn_start.pack(pady=20)

btn_exit = tk.Button(root, text="Exit", command=stop_app)
btn_exit.pack(pady=20)

root.mainloop()