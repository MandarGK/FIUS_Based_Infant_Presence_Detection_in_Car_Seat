import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from io import BytesIO
import os
import sys
import subprocess

# --------------------------
# Globals
# --------------------------
fft_img_tks = []
cm_img_tk = None
cm_image_label = None

# --------------------------
# Logging utility
# --------------------------
def log(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

# --------------------------
# FFT plotting
# --------------------------
def plot_fft_from_numpy(npy_paths):
    global fft_img_tks
    fft_img_tks.clear()

    # Clear previous widgets
    for widget in fft_scrollable_frame.winfo_children():
        widget.destroy()

    for npy_path in npy_paths:
        if os.path.exists(npy_path):
            try:
                fft_data = np.load(npy_path)
                freq = fft_data[:,0]       # Frequency
                magnitude = fft_data[:,1]  # FFT Magnitude

                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(freq, magnitude, color='blue')
                ax.set_title(os.path.basename(npy_path))
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.grid(True)

                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                img_tk = ImageTk.PhotoImage(img)
                fft_img_tks.append(img_tk)

                lbl = tk.Label(fft_scrollable_frame, image=img_tk)
                lbl.pack(pady=5)

                plt.close(fig)
            except Exception as e:
                log(f"Error plotting {npy_path}: {e}")

# --------------------------
# Confusion matrix
# --------------------------
def update_confusion_matrix(cm_path):
    global cm_img_tk, cm_image_label
    if os.path.exists(cm_path):
        img = Image.open(cm_path).resize((500,400))
        cm_img_tk = ImageTk.PhotoImage(img)
        if cm_image_label is None:
            cm_image_label = tk.Label(cm_tab, image=cm_img_tk)
            cm_image_label.pack(pady=10)
        else:
            cm_image_label.config(image=cm_img_tk)

# --------------------------
# Notebook execution
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
# Task pipeline
# --------------------------
def run_task_pipeline(task):
    # ----- ADC → FFT Conversion -----
    conversion_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Exploration"
    conversion_notebooks_map = {
        "Task1": ["ADC to FFT Emptyseat.ipynb", "ADC to FFT Carrierseat.ipynb"],
        "Task2": ["ADC to FFT Withbaby.ipynb"],
        "Task3": ["ADC to FFT Blanket and Sunscreen.ipynb"]
    }
    notebooks_to_run = conversion_notebooks_map.get(task, [])
    execute_notebooks(conversion_notebooks_path, notebooks_to_run)
    log("ADC → FFT conversion finished ✅")

    # Load FFT plots from folder dynamically
    fft_npy_paths = [
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Processed/Emptyseat_npy_array_Lowpassfiltered_label.npy",
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Processed/CarrierSeat_withoutBaby_Lowpassfilered_Label_0.npy"
    ]
    root.after(500, lambda: plot_fft_from_numpy(fft_npy_paths))

    # ----- Training Notebooks -----
    training_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Training"
    training_notebooks_map = {
        "Task1": ["Task1_Empty seat or carrier seat Classification using XG Boost model and MLP model.ipynb"],
        "Task2": ["Task2_With or without baby Detection using RanFor model and SVM model.ipynb"],
        "Task3": ["Task3_Baby presence detection when covered in blanket or sunscreen.ipynb"]
    }
    training_notebooks = training_notebooks_map.get(task, [])
    execute_notebooks(training_notebooks_path, training_notebooks)
    log("Notebook-based model training finished ✅")

    # Load Confusion Matrix image
    cm_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Images/confusion_matrix_Task1.png"
    root.after(500, lambda: update_confusion_matrix(cm_path))

# --------------------------
# Task window
# --------------------------
def run_task_window(task_name):
    task_win = tk.Toplevel(root)
    task_win.title(f"{task_name} - Train/Test Model")
    task_win.geometry("800x600")

    tk.Label(task_win, text=f"{task_name}: Train/Test ML Model", font=("Helvetica", 16, "bold")).pack(pady=10)

    global output_box
    output_box = scrolledtext.ScrolledText(task_win, width=95, height=20)
    output_box.pack(pady=10)

    def start_task():
        log(f"Starting {task_name}...")
        progress["mode"] = "indeterminate"
        progress.start(10)
        main_status.set(f"{task_name}: Running...")

        def task_thread():
            try:
                run_task_pipeline(task_name)
                progress.stop()
                progress["mode"] = "determinate"
                progress["value"] = 100
                main_status.set(f"{task_name}: Completed ✅")
                log("Task finished ✅")
            except Exception as e:
                progress.stop()
                main_status.set(f"{task_name}: Error ❌")
                log(str(e))
            finally:
                task1_btn.config(state="normal")
                task2_btn.config(state="normal")
                task3_btn.config(state="normal")

        # Disable buttons while running
        task1_btn.config(state="disabled")
        task2_btn.config(state="disabled")
        task3_btn.config(state="disabled")

        threading.Thread(target=task_thread, daemon=True).start()

    tk.Button(task_win, text="Start Task", width=25, command=start_task).pack(pady=10)
    tk.Button(task_win, text="Close", width=25, command=task_win.destroy).pack(pady=5)

# --------------------------
# Main GUI
# --------------------------
root = tk.Tk()
root.title("FIUS-Based Infant Presence Detection")
root.geometry("1000x700")

tk.Label(root, text="FIUS-Based Infant Presence Detection", font=("Helvetica", 18, "bold")).pack(pady=20)

frame = tk.Frame(root)
frame.pack(pady=10)

# Task buttons
task1_btn = tk.Button(frame, text="Task 1", width=20, command=lambda: run_task_window("Task1"))
task1_btn.grid(row=0, column=0, padx=20, pady=10)
task2_btn = tk.Button(frame, text="Task 2", width=20, command=lambda: run_task_window("Task2"))
task2_btn.grid(row=0, column=1, padx=20, pady=10)
task3_btn = tk.Button(frame, text="Task 3", width=20, command=lambda: run_task_window("Task3"))
task3_btn.grid(row=0, column=2, padx=20, pady=10)

# Status and progress
main_status = tk.StringVar()
main_status.set("Ready")
status_label = tk.Label(root, textvariable=main_status, font=("Helvetica", 12))
status_label.pack(pady=5)

progress = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
progress.pack(pady=5)

# Notebook tabs
notebook = ttk.Notebook(root)
fft_tab = tk.Frame(notebook)
cm_tab = tk.Frame(notebook)

# Scrollable frame for FFT
fft_canvas = tk.Canvas(fft_tab)
fft_scrollbar = ttk.Scrollbar(fft_tab, orient="vertical", command=fft_canvas.yview)
fft_scrollable_frame = tk.Frame(fft_canvas)

# Bind mouse wheel
def _on_mousewheel(event):
    fft_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

fft_scrollable_frame.bind("<Enter>", lambda e: fft_canvas.bind_all("<MouseWheel>", _on_mousewheel))
fft_scrollable_frame.bind("<Leave>", lambda e: fft_canvas.unbind_all("<MouseWheel>"))

fft_scrollable_frame.bind(
    "<Configure>",
    lambda e: fft_canvas.configure(scrollregion=fft_canvas.bbox("all"))
)

fft_canvas.create_window((0,0), window=fft_scrollable_frame, anchor="nw")
fft_canvas.configure(yscrollcommand=fft_scrollbar.set)
fft_canvas.pack(side="left", fill="both", expand=True)
fft_scrollbar.pack(side="right", fill="y")

notebook.add(fft_tab, text="FFT Plots")
notebook.add(cm_tab, text="Confusion Matrix")
notebook.pack(expand=True, fill="both", pady=20)

root.mainloop()
