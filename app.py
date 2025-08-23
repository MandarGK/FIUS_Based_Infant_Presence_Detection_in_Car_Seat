#!/usr/bin/env python3
"""
app.py - FIUS-Based Infant Presence Detection GUI

Key features:
- Runs ADC->FFT conversion notebooks (papermill)
- Runs training notebooks (papermill)
- Renders FFT plots from saved numpy arrays off the main thread (matplotlib Agg),
  sends PNG bytes to main thread for Tk display (avoids Tk use in worker threads)
- After training, displays executed notebook cell outputs in the "Confusion Matrix"
  tab (reads executed notebook if present, otherwise runs papermill to create executed file)
- Robust thread/process tracking and safe shutdown to avoid Tk runtime errors on exit
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import numpy as np
import matplotlib
# Ensure matplotlib uses Agg backend (not tkinter) when rendering in worker threads
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
from io import BytesIO
import os
import sys
import subprocess
import nbformat
import papermill as pm
import time

# --------------------------
# Globals / state
# --------------------------
fft_img_tks = []        # keep references to PhotoImage so they don't get GC'd
fft_img_labels = []     # keep Label widgets so we can destroy them on shutdown
cm_output_box = None    # will hold the ScrolledText widget for notebook output
output_box = None       # per-task logging area in the task window
root = None

# Track worker threads and spawned subprocesses (for safe shutdown)
worker_threads = set()
spawned_processes = set()
app_closing = False
lock = threading.Lock()

# --------------------------
# Thread/process helpers
# --------------------------
def start_worker(target, *args, **kwargs):
    """Start a non-daemon worker thread and track it."""
    t = threading.Thread(target=target, args=args, kwargs=kwargs)
    t.start()
    with lock:
        worker_threads.add(t)

    def _watch(_t=t):
        _t.join()
        with lock:
            worker_threads.discard(_t)
    threading.Thread(target=_watch, daemon=True).start()
    return t

def register_process(p: subprocess.Popen):
    with lock:
        spawned_processes.add(p)

def unregister_process(p: subprocess.Popen):
    with lock:
        spawned_processes.discard(p)

# --------------------------
# Safe logging to task output box
# --------------------------
def log(msg: str):
    """Append message to the task output box safely from any thread."""
    if output_box is None or root is None:
        return
    def _do():
        try:
            if output_box.winfo_exists():
                output_box.insert(tk.END, msg + "\n")
                output_box.see(tk.END)
        except Exception:
            pass
    root.after(0, _do)

# --------------------------
# FFT plotting (worker -> main thread)
# --------------------------
def _render_fft_plots_on_main_thread(png_bytes_list):
    """Given a list of PNG bytes, render them as PhotoImage and add to scrollable frame."""
    global fft_img_tks, fft_img_labels
    # Clear prior labels & images
    for lbl in fft_img_labels:
        try:
            lbl.config(image="")
            lbl.destroy()
        except Exception:
            pass
    fft_img_labels.clear()
    fft_img_tks.clear()

    for png in png_bytes_list:
        try:
            img = Image.open(BytesIO(png))
            img_tk = ImageTk.PhotoImage(img)
            fft_img_tks.append(img_tk)  # keep reference (very important)
            lbl = tk.Label(fft_scrollable_frame, image=img_tk)
            lbl.pack(pady=5)
            fft_img_labels.append(lbl)
        except Exception as e:
            log(f"Error creating Tk image: {e}")

def plot_fft_from_numpy(npy_paths):
    """Render matplotlib figures (to PNG) in a worker thread and deliver to main thread."""
    def worker():
        if app_closing:
            return
        rendered = []
        for path in npy_paths:
            if app_closing:
                return
            if not os.path.exists(path):
                log(f"FFT numpy file not found: {path}")
                continue
            try:
                arr = np.load(path)
                # Expect arr shape like (N, 4) or (N, >=2)
                freq = arr[:, 0]
                mag  = arr[:, 1]
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(freq, mag)
                ax.set_title(os.path.basename(path))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.grid(True)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                rendered.append(buf.getvalue())
            except Exception as e:
                log(f"Error plotting {path}: {e}")
        if not app_closing:
            root.after(0, lambda: _render_fft_plots_on_main_thread(rendered))

    # clear current UI first
    root.after(0, lambda: [w.destroy() for w in fft_scrollable_frame.winfo_children()])
    start_worker(worker)

# --------------------------
# Notebook output display logic
# --------------------------
def show_notebook_output_in_tab(original_nb_path):
    """
    Ensure cm_output_box (ScrolledText) exists, then:
    - prefer to read the executed notebook (executed_<name>.ipynb) if present already
    - otherwise execute the original notebook via papermill to create executed_<name>.ipynb
    - parse the executed notebook and display every cell's outputs into the cm_output_box
    This function is safe to call from worker threads; UI modifications are marshalled to main thread.
    """
    global cm_output_box

    # Create the cm_output_box on main thread synchronously (we schedule and wait briefly)
    created = threading.Event()
    def _create_box():
        global cm_output_box
        # Clear existing
        for w in cm_tab.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        cm_output_box = scrolledtext.ScrolledText(cm_tab, width=120, height=35)
        cm_output_box.pack(padx=10, pady=10, fill="both", expand=True)
        created.set()
    root.after(0, _create_box)
    # Wait until box created (safe short sleep loop)
    for _ in range(200):  # up to ~2 seconds
        if created.wait(timeout=0.01):
            break

    def worker():
        try:
            if app_closing:
                return
            # determine executed notebook path
            dirname = os.path.dirname(original_nb_path)
            base = os.path.basename(original_nb_path)
            executed_name = "executed_" + base
            executed_path = os.path.join(dirname, executed_name)

            # If executed exists already, read it; otherwise, run papermill to create it
            if not os.path.exists(executed_path):
                # run papermill (subprocess) and stream logs into output_box
                cmd = [sys.executable, "-m", "papermill", original_nb_path, executed_path]
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    register_process(proc)
                except FileNotFoundError as e:
                    log(f"Failed to start papermill: {e}")
                    return

                # stream output lines
                for line in proc.stdout:
                    if app_closing:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        break
                    log(line.rstrip("\n"))
                proc.wait()
                unregister_process(proc)
                if proc.returncode != 0:
                    log(f"papermill returned code {proc.returncode}")
                    # still attempt to read file if it exists
            else:
                log(f"Found existing executed notebook: {executed_path}")

            if app_closing:
                return
            if not os.path.exists(executed_path):
                root.after(0, lambda: cm_output_box.insert("1.0", f"Executed notebook not found: {executed_path}\n"))
                return

            # Read executed notebook and collect outputs
            nb = nbformat.read(executed_path, as_version=4)
            all_text_lines = []
            for idx, cell in enumerate(nb.cells):
                # Skip non-important cells
                tags = cell.get('metadata', {}).get('tags', [])
                if 'log_cm' not in tags:
                    continue  # skip cell if not tagged

                all_text_lines.append(f"--- Cell {idx} ({cell.cell_type}) ---\n")
                if cell.cell_type == "markdown":
                    all_text_lines.append(cell.source + "\n")
                elif cell.cell_type == "code":
                    outputs = cell.get("outputs", [])
                if not outputs:
                    all_text_lines.append("[no outputs]\n")
                else:
                    for out in outputs:
                        otype = out.get("output_type", "")
                        if otype == "stream":
                            all_text_lines.append(out.get("text", "") + "\n")
                        elif otype in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            text = data.get("text/plain")
                            if text:
                                all_text_lines.append(str(text) + "\n")
                        elif otype == "error":
                            tb = out.get("traceback", [])
                            if tb:
                                all_text_lines.append("\n".join(tb) + "\n")
            all_text_lines.append("\n")
            all_text = "".join(all_text_lines)

            # Insert into cm_output_box on main thread
            def _insert():
                if cm_output_box and cm_output_box.winfo_exists():
                    try:
                        cm_output_box.delete("1.0", tk.END)
                        cm_output_box.insert("1.0", all_text)
                        cm_output_box.see(tk.END)
                    except Exception:
                        pass
            root.after(0, _insert)

        except Exception as e:
            def _err():
                if cm_output_box and cm_output_box.winfo_exists():
                    cm_output_box.insert("1.0", f"Error while producing notebook output: {e}\n")
            root.after(0, _err)

    start_worker(worker)

# --------------------------
# Execute notebooks helper (used for conversion/training)
# --------------------------
def execute_notebooks(notebooks_path, notebooks_list):
    """
    Execute a list of notebooks using papermill via subprocess, stream output to output_box/log.
    This function runs synchronously (should be called from a worker thread).
    """
    for nb in notebooks_list:
        if app_closing:
            return
        input_nb = os.path.join(notebooks_path, nb)
        output_nb = os.path.join(notebooks_path, "executed_" + nb)
        log(f"Running notebook: {nb} ...")
        cmd = [sys.executable, "-m", "papermill", input_nb, output_nb]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            register_process(proc)
        except FileNotFoundError as e:
            log(f"Failed to start papermill for {nb}: {e}")
            continue

        for line in proc.stdout:
            if app_closing:
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            log(line.rstrip("\n"))
        proc.wait()
        unregister_process(proc)
        if proc.returncode == 0:
            log(f"Finished notebook: {nb}")
        else:
            log(f"Error running {nb}, return code: {proc.returncode}")

# --------------------------
# Task pipeline (background worker)
# --------------------------
def run_task_pipeline(task):
    """
    Orchestrates:
      - conversion notebooks (papermill)
      - load FFT NP arrays and plot
      - training notebooks (papermill)
      - show outputs of executed training notebook in CM tab
    Should be called from a worker thread (we use start_worker).
    """
    if app_closing:
        return

    # 1) ADC -> FFT conversion notebooks
    conversion_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Exploration"
    conversion_map = {
        "Task1": ["ADC to FFT Emptyseat.ipynb", "ADC to FFT Carrierseat.ipynb"],
        "Task2": ["ADC to FFT Withbaby.ipynb"],
        "Task3": ["ADC to FFT Blanket and Sunscreen.ipynb"]
    }
    conv_list = conversion_map.get(task, [])
    if conv_list:
        execute_notebooks(conversion_notebooks_path, conv_list)
        log("ADC -> FFT conversion finished ✅")

    # 2) plot FFTs (from npy)
    fft_npy_paths = [
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Processed/Emptyseat_npy_array_Lowpassfiltered_label.npy",
        "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Data/Processed/CarrierSeat_withoutBaby_Lowpassfilered_Label_0.npy"
    ]
    plot_fft_from_numpy(fft_npy_paths)

    # 3) training notebooks
    training_notebooks_path = "/Users/mandarkale/Documents/MyProjects/MachineLearning/FIUS_Based_Infant_Presence_Detection_in_Car_Seats/source_code/Notebooks/Training"
    training_map = {
        "Task1": ["Task1_Empty seat or carrier seat Classification using XG Boost model and MLP model.ipynb"],
        "Task2": ["Task2_With or without baby Detection using RanFor model and SVM model.ipynb"],
        "Task3": ["Task3_Baby presence detection when covered in blanket or sunscreen.ipynb"]
    }
    train_list = training_map.get(task, [])
    if train_list:
        execute_notebooks(training_notebooks_path, train_list)
        log("Notebook-based model training finished ✅")

    # 4) show outputs of the executed training notebook (prefer executed_*.ipynb)
    if train_list:
        # pass the original notebook path (show_notebook_output_in_tab will prefer executed_ if present)
        original_nb = os.path.join(training_notebooks_path, train_list[0])
        show_notebook_output_in_tab(original_nb)

# --------------------------
# Task window GUI
# --------------------------
def run_task_window(task_name):
    task_win = tk.Toplevel(root)
    task_win.title(f"{task_name} - Train/Test Model")
    task_win.geometry("800x600")

    tk.Label(task_win, text=f"{task_name}: Train/Test ML Model",
             font=("Helvetica", 16, "bold")).pack(pady=10)

    global output_box
    output_box = scrolledtext.ScrolledText(task_win, width=95, height=20)
    output_box.pack(pady=10)

    start_btn = tk.Button(task_win, text="Start Task", width=25, disabledforeground="white")
    start_btn.pack(pady=10)
    tk.Button(task_win, text="Close", width=25, command=task_win.destroy).pack(pady=5)

    def _start_task():
        log(f"Starting {task_name}...")
        progress.config(mode="indeterminate")
        progress.start(8)
        main_status.set(f"{task_name}: Running...")

        # disable buttons
        start_btn.config(state="disabled")
        task1_btn.config(state="disabled")
        task2_btn.config(state="disabled")
        task3_btn.config(state="disabled")

        def worker():
            try:
                run_task_pipeline(task_name)
                # on success, update UI on main thread
                def _done():
                    progress.stop()
                    progress.config(mode="determinate", value=100)
                    main_status.set(f"{task_name}: Completed ✅")
                    log("Task finished ✅")
                root.after(0, _done)
            except Exception as e:
                def _err():
                    progress.stop()
                    main_status.set(f"{task_name}: Error ❌")
                    log(f"Task error: {e}")
                root.after(0, _err)
            finally:
                # re-enable buttons
                def _reenable():
                    start_btn.config(state="normal")
                    task1_btn.config(state="normal")
                    task2_btn.config(state="normal")
                    task3_btn.config(state="normal")
                root.after(0, _reenable)

        start_worker(worker)

    start_btn.config(command=_start_task)

# --------------------------
# Clean shutdown handler
# --------------------------
def on_close():
    """Attempt graceful shutdown: stop processes, join threads, clear images, then destroy root."""
    global app_closing
    app_closing = True

    with lock:
        procs = list(spawned_processes)
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    # wait a short time then kill if still alive
    deadline = time.time() + 3.0
    for p in procs:
        try:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    # join worker threads briefly
    with lock:
        threads = list(worker_threads)
    for t in threads:
        try:
            t.join(timeout=1.0)
        except Exception:
            pass

    # remove Tk images & widgets before destroying root to avoid Image.__del__ calling Tk from wrong thread
    for lbl in fft_img_labels:
        try:
            lbl.config(image="")
            lbl.destroy()
        except Exception:
            pass
    fft_img_labels.clear()
    fft_img_tks.clear()

    try:
        root.destroy()
    except Exception:
        pass

# --------------------------
# Main GUI build
# --------------------------
root = tk.Tk()
root.title("FIUS-Based Infant Presence Detection")
root.geometry("1000x700")
root.protocol("WM_DELETE_WINDOW", on_close)

tk.Label(root, text="FIUS-Based Infant Presence Detection",
         font=("Helvetica", 18, "bold")).pack(pady=20)

frame = tk.Frame(root)
frame.pack(pady=10)

# Task buttons
task1_btn = tk.Button(frame, text="Task 1", width=20,
                      command=lambda: run_task_window("Task1"),disabledforeground="white")
task1_btn.grid(row=0, column=0, padx=20, pady=10)

task2_btn = tk.Button(frame, text="Task 2", width=20,
                      command=lambda: run_task_window("Task2"),disabledforeground="white")
task2_btn.grid(row=0, column=1, padx=20, pady=10)

task3_btn = tk.Button(frame, text="Task 3", width=20,
                      command=lambda: run_task_window("Task3"),disabledforeground="white")
task3_btn.grid(row=0, column=2, padx=20, pady=10)

# Status and progress
main_status = tk.StringVar(master=root, value="Ready")
status_label = tk.Label(root, textvariable=main_status, font=("Helvetica", 12))
status_label.pack(pady=5)

progress = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
progress.pack(pady=5)

# Notebook: FFT plots + Confusion Matrix / outputs
notebook = ttk.Notebook(root)
fft_tab = tk.Frame(notebook)
cm_tab  = tk.Frame(notebook)

# build scrollable frame for FFT images
fft_canvas = tk.Canvas(fft_tab)
fft_scrollbar = ttk.Scrollbar(fft_tab, orient="vertical", command=fft_canvas.yview)
fft_scrollable_frame = tk.Frame(fft_canvas)

def _on_mousewheel(event):
    fft_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
fft_scrollable_frame.bind("<Enter>", lambda e: fft_canvas.bind_all("<MouseWheel>", _on_mousewheel))
fft_scrollable_frame.bind("<Leave>", lambda e: fft_canvas.unbind_all("<MouseWheel>"))

fft_scrollable_frame.bind("<Configure>", lambda e: fft_canvas.configure(scrollregion=fft_canvas.bbox("all")))

fft_canvas.create_window((0, 0), window=fft_scrollable_frame, anchor="nw")
fft_canvas.configure(yscrollcommand=fft_scrollbar.set)
fft_canvas.pack(side="left", fill="both", expand=True)
fft_scrollbar.pack(side="right", fill="y")

notebook.add(fft_tab, text="FFT Plots")
notebook.add(cm_tab, text="Confusion Matrix / Notebook Outputs")
notebook.pack(expand=True, fill="both", pady=20)

root.mainloop()
