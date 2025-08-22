import tkinter as tk
from tkinter import scrolledtext, ttk
import os
import subprocess
import threading
import sys
import time
from PIL import Image, ImageTk

# --------------------------
# Globals for Images
# --------------------------
fft_img_tks = []       # To prevent GC for FFT plots
cm_img_tk = None       # To prevent GC for CM
cm_image_label = None

# --------------------------
# GUI Utility
# --------------------------
def log(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

# Update FFT tab with multiple FFT plots
def update_fft_images(image_paths):
    global fft_img_tks
    fft_img_tks.clear()
    for widget in fft_tab.winfo_children():
        widget.destroy()

    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((500, 300))
            img_tk = ImageTk.PhotoImage(img)
            fft_img_tks.append(img_tk)

            lbl = tk.Label(fft_tab, image=img_tk)
            lbl.pack(pady=5)

            # Add label name under plot
            tk.Label(fft_tab, text=os.path.basename(img_path),
                     font=("Helvetica", 10, "italic")).pack(pady=2)

# Update Confusion Matrix tab
def update_confusion_matrix(cm_path):
    global cm_img_tk, cm_image_label
    if os.path.exists(cm_path):
        img = Image.open(cm_path)
        img = img.resize((500, 400))
        cm_img_tk = ImageTk.PhotoImage(img)
        if cm_image_label is None:
            cm_image_label = tk.Label(cm_tab, image=cm_img_tk)
            cm_image_label.pack(pady=10)
        else:
            cm_image_label.config(image=cm_img_tk)

# --------------------------
# Notebook Execution Functions
# --------------------------
def execute_notebooks(notebooks_path, notebooks_list):
    for nb in notebooks_list:
        input_nb = os.path.join(notebooks_path, nb)
        output_nb = os.path.join(notebooks_path, "executed_" + nb)
        log(f"Running notebook: {nb} ...")
        cmd = [sys.executable, "-m", "papermill", input_nb, output_nb]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log(line.strip())
        process.wait()
        if process.returncode == 0:
            log(f"Finished notebook: {nb}")
        else:
            log(f"Error running {nb}, return code: {process.returncode}")

# --------------------------
# Data Conversion + Training Dispatcher
# --------------------------
def run_task_pipeline(task):
    # 1. ADC → FFT conversion notebooks
    conversion_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Exploration"
    conversion_notebooks_map = {
        "Task1": ["ADC to FFT Emptyseat.ipynb", "ADC to FFT Carrierseat.ipynb"],
        "Task2": ["ADC to FFT Withbaby.ipynb"],
        "Task3": ["ADC to FFT Blanket and Sunscreen.ipynb"]
    }
    notebooks_to_run = conversion_notebooks_map.get(task, [])
    execute_notebooks(conversion_notebooks_path, notebooks_to_run)
    log("ADC → FFT conversion finished ✅")

    # Update FFT images in the main window
    fft_plot_paths = [
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Images/Figure_plot_Task1_Empty.png",
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Images/Figure_plot_Task1_Carrier.png"
    ]
    root.after(500, lambda: update_fft_images(fft_plot_paths))

    # 2. Training notebooks
    training_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Training"
    training_notebooks_map = {
        "Task1": ["Task1_Empty seat or carrier seat Classification using XG Boost model and MLP model.ipynb"],
        "Task2": ["Task2_With or without baby Detection using RanFor model and SVM model.ipynb"],
        "Task3": ["Task3_Baby presence detection when covered in blanket or sunscreen.ipynb"]
    }
    training_notebooks = training_notebooks_map.get(task, [])
    execute_notebooks(training_notebooks_path, training_notebooks)
    log("Notebook-based model training finished ✅")

    # Update Confusion Matrix image
    cm_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Images/confusion_matrix_Task1.png"
    root.after(500, lambda: update_confusion_matrix(cm_path))

# --------------------------
# Task Window
# --------------------------
def run_task_window(task_name):
    task_win = tk.Toplevel(root)
    task_win.title(f"{task_name} - Train/Test Model")
    task_win.geometry("750x550")

    tk.Label(task_win, text=f"{task_name}: Train/Test ML Model", font=("Helvetica", 16, "bold")).pack(pady=10)

    global output_box
    output_box = scrolledtext.ScrolledText(task_win, width=90, height=25)
    output_box.pack(pady=10)

    def training_process():
        log("Starting training process using Papermill...")
        log("Please wait... Training in progress...")

        # Disable main window task buttons
        task1_btn.config(state="disabled")
        task2_btn.config(state="disabled")
        task3_btn.config(state="disabled")

        # Start progress bar in indeterminate mode
        progress["mode"] = "indeterminate"
        progress.start(10)  # adjust speed for left-right movement
        main_status.set(f"{task_name}: Running...")

        def run_in_thread():
            try:
                run_task_pipeline(task_name)
                main_status.set(f"{task_name}: Completed ✅")
                log("All tasks finished ✅")
                # Stop indeterminate and turn bar fully blue
                progress.stop()
                progress["mode"] = "determinate"
                progress["value"] = 100
            except Exception as e:
                main_status.set(f"{task_name}: Error ❌")
                log(f"Error during training: {e}")
                progress.stop()
                progress["mode"] = "determinate"
                progress["value"] = 0
            finally:
                task1_btn.config(state="normal")
                task2_btn.config(state="normal")
                task3_btn.config(state="normal")


        threading.Thread(target=run_in_thread, daemon=True).start()

    tk.Button(task_win, text="Start Training", width=25, command=training_process).pack(pady=10)
    tk.Button(task_win, text="Close", width=25, command=task_win.destroy).pack(pady=5)

# --------------------------
# Main GUI
# --------------------------
root = tk.Tk()
root.title("FIUS-Based Infant Presence Detection")
root.geometry("1000x700")

tk.Label(root, text="FIUS-Based Infant Presence Detection in Car Seats", font=("Helvetica", 18, "bold")).pack(pady=20)

frame = tk.Frame(root)
frame.pack(pady=10)

# Task Buttons
task1_btn = tk.Button(frame, text="Task 1", width=20, command=lambda: run_task_window("Task1"), disabledforeground="white")
task1_btn.grid(row=0, column=0, padx=20, pady=10)
task2_btn = tk.Button(frame, text="Task 2", width=20, command=lambda: run_task_window("Task2"), disabledforeground="white")
task2_btn.grid(row=0, column=1, padx=20, pady=10)
task3_btn = tk.Button(frame, text="Task 3", width=20, command=lambda: run_task_window("Task3"), disabledforeground="white")
task3_btn.grid(row=0, column=2, padx=20, pady=10)

# Status Label
main_status = tk.StringVar()
main_status.set("Ready")
status_label = tk.Label(root, textvariable=main_status, font=("Helvetica", 12))
status_label.pack(pady=5)

# Progress Bar (independent, left-right moving)
progress = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
progress.pack(pady=5)

# Notebook Tabs for FFT + Confusion Matrix
notebook = ttk.Notebook(root)
fft_tab = tk.Frame(notebook)
cm_tab = tk.Frame(notebook)

notebook.add(fft_tab, text="FFT Plots")
notebook.add(cm_tab, text="Confusion Matrix")
notebook.pack(expand=True, fill="both", pady=20)

root.mainloop()
