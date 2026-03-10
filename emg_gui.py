"""
EMG Data Collection GUI
=======================
A modern GUI for the EMG data collection pipeline.

Features:
  - Data collection with live EMG visualization and gesture prompts
  - Session inspector with signal and feature plots
  - Model training with progress and results
  - Live prediction demo
  - LDA visualization

Requirements:
  pip install customtkinter matplotlib numpy

Run:
  python emg_gui.py
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time
import sys
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Import from the existing pipeline
from learning_data_collection import (
    # Configuration
    NUM_CHANNELS, SAMPLING_RATE_HZ, WINDOW_SIZE_MS, HOP_SIZE_MS, HAND_CHANNELS,
    GESTURE_HOLD_SEC, REST_BETWEEN_SEC, REPS_PER_GESTURE, DATA_DIR, MODEL_DIR, USER_ID,
    # Classes
    EMGSample, EMGWindow, EMGParser, Windower,
    PromptScheduler, SessionStorage, SessionMetadata,
    EMGFeatureExtractor, EMGClassifier, PredictionSmoother, CalibrationTransform,
    LABEL_SHIFT_MS,
)

# Import real serial stream for ESP32 hardware
from serial_stream import RealSerialStream
import serial.tools.list_ports

# =============================================================================
# APPEARANCE SETTINGS
# =============================================================================

ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Colors for gestures (names match ESP32 gesture definitions)
GESTURE_COLORS = {
    "rest": "#6c757d",        # Gray
    "open": "#17a2b8",        # Cyan
    "fist": "#007bff",        # Blue
    "hook_em": "#fd7e14",     # Orange (Hook 'em Horns)
    "thumbs_up": "#28a745",   # Green
}

CALIB_PREP_SEC = 3         # Seconds of "get ready" countdown before each gesture
CALIB_DURATION_SEC = 5.0   # Seconds to hold each gesture during calibration


def get_gesture_color(gesture_name: str) -> str:
    """Get color for a gesture name."""
    for key, color in GESTURE_COLORS.items():
        if key in gesture_name.lower():
            return color
    return "#dc3545"  # Red for unknown


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class EMGApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("EMG Data Collection Pipeline")
        self.geometry("1400x900")
        self.minsize(1200, 700)

        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar
        self.sidebar = Sidebar(self, self.show_page)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Create container for pages
        self.page_container = ctk.CTkFrame(self, fg_color="transparent")
        self.page_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        # Calibrated classifier shared between CalibrationPage and PredictionPage.
        # Set by CalibrationPage._apply_calibration(), read by PredictionPage.
        self.calibrated_classifier = None

        # Create pages
        self.pages = {}
        self.pages["collection"] = CollectionPage(self.page_container)
        self.pages["inspect"] = InspectPage(self.page_container)
        self.pages["training"] = TrainingPage(self.page_container)
        self.pages["calibration"] = CalibrationPage(self.page_container)
        self.pages["prediction"] = PredictionPage(self.page_container)
        self.pages["visualization"] = VisualizationPage(self.page_container)

        # Show default page
        self.current_page = None
        self.show_page("collection")

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_page(self, page_name: str):
        """Show a specific page."""
        # Hide current page
        if self.current_page:
            self.pages[self.current_page].grid_forget()
            self.pages[self.current_page].on_hide()

        # Show new page
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        self.pages[page_name].on_show()
        self.current_page = page_name

        # Update sidebar selection
        self.sidebar.set_active(page_name)

    def on_close(self):
        """Handle window close."""
        # Stop any running processes in pages
        for page in self.pages.values():
            if hasattr(page, 'stop'):
                page.stop()
        self.destroy()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

class Sidebar(ctk.CTkFrame):
    """Sidebar navigation panel."""

    def __init__(self, parent, on_select_callback):
        super().__init__(parent, width=200, corner_radius=0)
        self.on_select = on_select_callback

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self, text="EMG Pipeline",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.pack(pady=(20, 10))

        self.subtitle = ctk.CTkLabel(
            self, text="Data Collection & ML",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.subtitle.pack(pady=(0, 20))

        # Navigation buttons
        self.nav_buttons = {}

        nav_items = [
            ("collection", "1. Collect Data"),
            ("inspect", "2. Inspect Sessions"),
            ("training", "3. Train Model"),
            ("calibration", "4. Calibrate"),
            ("prediction", "5. Live Prediction"),
            ("visualization", "6. Visualize LDA"),
        ]

        for page_id, label in nav_items:
            btn = ctk.CTkButton(
                self, text=label,
                font=ctk.CTkFont(size=14),
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                anchor="w",
                command=lambda p=page_id: self.on_select(p)
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.nav_buttons[page_id] = btn

        # Spacer
        spacer = ctk.CTkLabel(self, text="")
        spacer.pack(expand=True)

        # Status area
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(fill="x", padx=10, pady=10)

        self.session_count_label = ctk.CTkLabel(
            self.status_frame, text="Sessions: 0",
            font=ctk.CTkFont(size=12)
        )
        self.session_count_label.pack()

        self.model_status_label = ctk.CTkLabel(
            self.status_frame, text="Model: Not saved",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.model_status_label.pack()

        # Update status
        self.update_status()

    def set_active(self, page_id: str):
        """Set the active navigation button."""
        for pid, btn in self.nav_buttons.items():
            if pid == page_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

    def update_status(self):
        """Update the status display."""
        storage = SessionStorage()
        sessions = storage.list_sessions()
        self.session_count_label.configure(text=f"Sessions: {len(sessions)}")

        model_path = EMGClassifier.get_latest_model_path()
        if model_path:
            self.model_status_label.configure(text=f"Model: {model_path.stem}", text_color="green")
        else:
            self.model_status_label.configure(text="Model: Not saved", text_color="gray")


# =============================================================================
# BASE PAGE CLASS
# =============================================================================

class BasePage(ctk.CTkFrame):
    """Base class for all pages."""

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def on_show(self):
        """Called when page is shown."""
        pass

    def on_hide(self):
        """Called when page is hidden."""
        pass

    def create_header(self, title: str, subtitle: str = ""):
        """Create a page header."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        title_label = ctk.CTkLabel(
            header_frame, text=title,
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(anchor="w")

        if subtitle:
            subtitle_label = ctk.CTkLabel(
                header_frame, text=subtitle,
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            subtitle_label.pack(anchor="w")

        return header_frame


# =============================================================================
# DATA COLLECTION PAGE
# =============================================================================

class CollectionPage(BasePage):
    """Data collection page with live EMG visualization."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Data Collection",
            "Collect labeled EMG data with timed gesture prompts"
        )

        # Collection state (MUST be initialized BEFORE setup_controls)
        self.is_collecting = False
        self.is_connected = False
        self.using_real_hardware = True  # Always use real ESP32 hardware
        self.stream = None
        self.parser = None
        self.windower = None
        self.scheduler = None
        self.collected_windows = []
        self.collected_labels = []
        self.collected_raw_samples = []  # For label alignment
        self.sample_buffer = []
        self.collection_thread = None
        self.data_queue = queue.Queue()

        # Main content area
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(0, weight=1)

        # Left panel - Controls
        self.controls_panel = ctk.CTkFrame(self.content)
        self.controls_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.setup_controls()

        # Right panel - Live plot and prompt
        self.plot_panel = ctk.CTkFrame(self.content)
        self.plot_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.setup_plot()

    def setup_controls(self):
        """Setup the control panel."""
        # User ID
        user_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        user_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(user_frame, text="User ID:", font=ctk.CTkFont(size=14)).pack(anchor="w")
        self.user_id_entry = ctk.CTkEntry(user_frame, placeholder_text="user_001")
        self.user_id_entry.pack(fill="x", pady=(5, 0))
        self.user_id_entry.insert(0, USER_ID)

        # ESP32 Connection (hardware required)
        source_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        source_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(source_frame, text="ESP32 Connection:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        # Port selection
        port_select_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        port_select_frame.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(port_select_frame, text="Port:").pack(side="left")

        self.port_var = ctk.StringVar(value="Auto-detect")
        self.port_dropdown = ctk.CTkOptionMenu(
            port_select_frame, variable=self.port_var,
            values=["Auto-detect"], width=150
        )
        self.port_dropdown.pack(side="left", padx=(10, 5))

        self.refresh_ports_btn = ctk.CTkButton(
            port_select_frame, text="⟳", width=30,
            command=self._refresh_ports
        )
        self.refresh_ports_btn.pack(side="left")

        # Connection status and button
        connect_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        connect_frame.pack(fill="x", pady=(5, 0))

        self.connect_button = ctk.CTkButton(
            connect_frame, text="Connect",
            width=100, height=28,
            command=self._toggle_connection
        )
        self.connect_button.pack(side="left", padx=(0, 10))

        self.connection_status = ctk.CTkLabel(
            connect_frame, text="● Disconnected",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.connection_status.pack(side="left")

        # Refresh ports on startup
        self._refresh_ports()

        # Gesture selection
        gesture_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        gesture_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(gesture_frame, text="Gestures:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        self.gesture_vars = {}
        available_gestures = ["open", "fist", "hook_em", "thumbs_up"]

        for gesture in available_gestures:
            var = ctk.BooleanVar(value=True)  # All selected by default
            cb = ctk.CTkCheckBox(gesture_frame, text=gesture.replace("_", " ").title(), variable=var)
            cb.pack(anchor="w", pady=2)
            self.gesture_vars[gesture] = var

        # Settings
        settings_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        settings_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(settings_frame, text="Settings:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        # Hold duration
        hold_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        hold_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(hold_frame, text="Hold (sec):").pack(side="left")
        self.hold_slider = ctk.CTkSlider(hold_frame, from_=1, to=5, number_of_steps=8)
        self.hold_slider.set(GESTURE_HOLD_SEC)
        self.hold_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.hold_label = ctk.CTkLabel(hold_frame, text=f"{GESTURE_HOLD_SEC:.1f}")
        self.hold_label.pack(side="right")
        self.hold_slider.configure(command=lambda v: self.hold_label.configure(text=f"{v:.1f}"))

        # Reps
        reps_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        reps_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(reps_frame, text="Reps:").pack(side="left")
        self.reps_slider = ctk.CTkSlider(reps_frame, from_=1, to=5, number_of_steps=4)
        self.reps_slider.set(REPS_PER_GESTURE)
        self.reps_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.reps_label = ctk.CTkLabel(reps_frame, text=f"{REPS_PER_GESTURE}")
        self.reps_label.pack(side="right")
        self.reps_slider.configure(command=lambda v: self.reps_label.configure(text=f"{int(v)}"))

        # Buttons
        button_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)

        self.start_button = ctk.CTkButton(
            button_frame, text="Start Collection",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.toggle_collection
        )
        self.start_button.pack(fill="x", pady=5)

        self.save_button = ctk.CTkButton(
            button_frame, text="Save Session",
            font=ctk.CTkFont(size=14),
            height=40,
            state="disabled",
            command=self.save_session
        )
        self.save_button.pack(fill="x", pady=5)

        # Progress
        progress_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=10)

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            progress_frame, text="Ready to collect",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack()

        self.window_count_label = ctk.CTkLabel(
            progress_frame, text="Windows: 0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.window_count_label.pack()

    def setup_plot(self):
        """Setup the live plot area."""
        # Gesture prompt display
        self.prompt_frame = ctk.CTkFrame(self.plot_panel)
        self.prompt_frame.pack(fill="x", padx=20, pady=20)

        self.prompt_label = ctk.CTkLabel(
            self.prompt_frame, text="READY",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color="gray",
            width=500,  # Fixed width to prevent resizing glitches
        )
        self.prompt_label.pack(pady=30)

        self.countdown_label = ctk.CTkLabel(
            self.prompt_frame, text="",
            font=ctk.CTkFont(size=18)
        )
        self.countdown_label.pack()

        # Matplotlib figure for live EMG
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#2b2b2b')
        self.axes = []

        for i in range(NUM_CHANNELS):
            ax = self.fig.add_subplot(NUM_CHANNELS, 1, i + 1)
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.set_ylabel(f'Ch{i}', color='white', fontsize=10)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 3300)  # ESP32 outputs millivolts (0-3100 mV)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('white')
            self.axes.append(ax)

        self.axes[-1].set_xlabel('Samples', color='white')
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Initialize plot lines
        self.plot_lines = []
        self.plot_data = [np.zeros(500) for _ in range(NUM_CHANNELS)]

        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.plot_data[i], color='#00ff88', linewidth=1)
            self.plot_lines.append(line)

    def toggle_collection(self):
        """Start or stop collection."""
        print("\n" + "="*80)
        print("[DEBUG] toggle_collection() called")
        print(f"[DEBUG] Current state:")
        print(f"  - is_collecting: {self.is_collecting}")
        print(f"  - is_connected: {self.is_connected}")
        print(f"  - stream exists: {self.stream is not None}")
        if self.stream:
            if hasattr(self.stream, 'state'):
                print(f"  - stream.state: {self.stream.state}")
        print(f"  - button text: {self.start_button.cget('text')}")
        print(f"  - button state: {self.start_button.cget('state')}")

        # Prevent rapid double-clicks from interfering
        if hasattr(self, '_toggling') and self._toggling:
            print("[DEBUG] BLOCKED: Already toggling (debounce)")
            print("="*80 + "\n")
            return

        self._toggling = True
        try:
            if self.is_collecting:
                print("[DEBUG] Branch: STOPPING collection")
                self.stop_collection()
            else:
                print("[DEBUG] Branch: STARTING collection")
                self.start_collection()
        finally:
            # Reset flag after brief delay to prevent immediate re-trigger
            self.after(100, lambda: setattr(self, '_toggling', False))

    def start_collection(self):
        """Start data collection."""
        print("[DEBUG] start_collection() entered")

        # CRITICAL: Drain any stale messages from previous sessions FIRST
        # This prevents old 'done' messages from stopping the new session
        stale_count = 0
        try:
            while True:
                msg = self.data_queue.get_nowait()
                stale_count += 1
                print(f"[DEBUG] Drained stale message: {msg[0]}")
        except queue.Empty:
            pass
        if stale_count > 0:
            print(f"[DEBUG] Cleared {stale_count} stale message(s) from queue")

        # Get selected gestures
        gestures = [g for g, var in self.gesture_vars.items() if var.get()]
        print(f"[DEBUG] Selected gestures: {gestures}")
        if not gestures:
            print("[DEBUG] EXIT: No gestures selected")
            messagebox.showwarning("No Gestures", "Please select at least one gesture.")
            return

        # Must be connected to ESP32
        print(f"[DEBUG] Checking connection: is_connected={self.is_connected}, stream exists={self.stream is not None}")
        if not self.is_connected or not self.stream:
            print("[DEBUG] EXIT: Not connected to device")
            messagebox.showerror("Not Connected", "Please connect to the ESP32 first.")
            return

        # Send start command to begin streaming
        print("[DEBUG] Calling stream.start()...")
        try:
            self.stream.start()
            print("[DEBUG] stream.start() succeeded")
        except Exception as e:
            print(f"[DEBUG] stream.start() FAILED: {e}")
            # Reset stream state if start failed
            if self.stream:
                try:
                    print("[DEBUG] Attempting stream.stop() to reset state...")
                    self.stream.stop()  # Try to return to CONNECTED state
                    print("[DEBUG] stream.stop() succeeded")
                except Exception as e2:
                    print(f"[DEBUG] stream.stop() FAILED: {e2}")
            messagebox.showerror("Start Error", f"Failed to start streaming:\n{e}")
            print("[DEBUG] EXIT: Stream start error")
            return

        # Initialize parser and windower
        self.parser = EMGParser(num_channels=NUM_CHANNELS)
        self.windower = Windower(
            window_size_ms=WINDOW_SIZE_MS,
            sample_rate=SAMPLING_RATE_HZ,
            hop_size_ms=HOP_SIZE_MS
        )

        self.scheduler = PromptScheduler(
            gestures=gestures,
            hold_sec=self.hold_slider.get(),
            rest_sec=REST_BETWEEN_SEC,
            reps=int(self.reps_slider.get())
        )

        # Reset state
        self.collected_windows = []
        self.collected_labels = []
        self.collected_trial_ids = []  # Track trial_ids for proper train/test splitting
        self.collected_raw_samples = []  # Store raw samples for label alignment
        self.sample_buffer = []
        print("[DEBUG] Reset collection state")

        # Mark as collecting
        self.is_collecting = True
        print("[DEBUG] Set is_collecting = True")

        # Update UI
        self.start_button.configure(text="Stop Collection", fg_color="red")
        self.save_button.configure(state="disabled")
        self.status_label.configure(text="Starting...")
        print("[DEBUG] Updated UI - button now shows 'Stop Collection'")

        # Disable connection controls during collection
        self.connect_button.configure(state="disabled")
        print("[DEBUG] Disabled connection controls")

        # Start collection thread
        self.collection_thread = threading.Thread(target=self.collection_loop, daemon=True)
        self.collection_thread.start()
        print("[DEBUG] Started collection thread")

        # Start UI update loop
        self.update_collection_ui()
        print("[DEBUG] start_collection() completed successfully")
        print("="*80 + "\n")

    def stop_collection(self):
        """Stop data collection."""
        print("[DEBUG] stop_collection() called")
        print(f"[DEBUG] Was collecting: {self.is_collecting}")
        self.is_collecting = False

        # Safe cleanup - stream might already be in error state
        try:
            if self.stream:
                print("[DEBUG] Calling stream.stop()")
                # Send stop command (returns to CONNECTED state)
                self.stream.stop()
                print("[DEBUG] stream.stop() completed")
        except Exception as e:
            print(f"[DEBUG] Exception during stream cleanup: {e}")
            pass  # Ignore cleanup errors

        # Drain any pending messages from queue to prevent stale data
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass

        self.start_button.configure(text="Start Collection", fg_color=["#3B8ED0", "#1F6AA5"])
        self.status_label.configure(text=f"Collected {len(self.collected_windows)} windows")
        self.prompt_label.configure(text="DONE", text_color="green")
        self.countdown_label.configure(text="")
        print("[DEBUG] UI reset - button shows 'Start Collection'")

        # Re-enable connection button
        self.connect_button.configure(state="normal")
        # Still connected, just not streaming
        if self.is_connected:
            device_name = self.stream.device_info.get('device', 'ESP32') if self.stream and self.stream.device_info else 'ESP32'
            self._update_connection_status("green", f"Connected ({device_name})")

        if self.collected_windows:
            self.save_button.configure(state="normal")

        print("[DEBUG] stop_collection() completed")
        print("="*80 + "\n")

    def collection_loop(self):
        """Background collection loop."""
        # Stream is already started via handshake
        self.data_queue.put(('connection_status', ('green', 'Streaming')))

        self.scheduler.start_session()

        last_prompt = None
        last_ui_update = time.perf_counter()
        last_plot_update = time.perf_counter()
        last_data_time = time.perf_counter()  # Track last received data for timeout detection
        sample_batch = []  # Batch samples for plotting
        timeout_warning_sent = False

        while self.is_collecting and not self.scheduler.is_session_complete():
            # Get current prompt
            prompt = self.scheduler.get_current_prompt()
            current_time = time.perf_counter()

            if prompt:
                # Calculate time remaining in current gesture
                elapsed_in_session = self.scheduler.get_elapsed_time()
                elapsed_in_gesture = elapsed_in_session - prompt.start_time
                time_remaining_in_gesture = prompt.duration_sec - elapsed_in_gesture

                # Find the next gesture (for "upcoming" display)
                current_prompt_idx = self.scheduler.schedule.prompts.index(prompt)
                next_gesture = None
                if current_prompt_idx + 1 < len(self.scheduler.schedule.prompts):
                    next_prompt = self.scheduler.schedule.prompts[current_prompt_idx + 1]
                    if next_prompt.gesture_name != "rest":
                        next_gesture = next_prompt.gesture_name

                # Send prompt update to UI (throttled to every 200ms for smoother text)
                if current_time - last_ui_update > 0.2:
                    # Send current gesture, countdown, and upcoming gesture
                    self.data_queue.put(('prompt_with_countdown', (
                        prompt.gesture_name,
                        time_remaining_in_gesture,
                        next_gesture
                    )))

                    # Send overall progress
                    progress = elapsed_in_session / self.scheduler.schedule.total_duration
                    self.data_queue.put(('progress', progress))
                    last_ui_update = current_time

                    last_prompt = prompt.gesture_name

            # Read and process data
            try:
                line = self.stream.readline()
            except Exception as e:
                # Only report error if we didn't intentionally stop
                if self.is_collecting:
                    self.data_queue.put(('error', f"Serial read error: {e}"))
                break

            if line:
                last_data_time = current_time  # Reset timeout counter
                timeout_warning_sent = False
                sample = self.parser.parse_line(line)
                if sample:
                    # Store raw sample for label alignment
                    self.collected_raw_samples.append(sample)

                    # Batch samples for plotting (don't send every single one)
                    sample_batch.append(sample.channels)

                    # Send batched samples for plotting every 50ms (20 FPS)
                    if current_time - last_plot_update > 0.05:
                        if sample_batch:
                            self.data_queue.put(('samples_batch', sample_batch))
                            sample_batch = []
                            last_plot_update = current_time

                    # Try to form a window
                    window = self.windower.add_sample(sample)
                    if window:
                        # Shift label lookup forward to align with actual muscle
                        # activation (accounts for reaction time + window centre)
                        label_time = window.start_time + LABEL_SHIFT_MS / 1000.0
                        label = self.scheduler.get_label_for_time(label_time)
                        trial_id = self.scheduler.get_trial_id_for_time(label_time)
                        self.collected_windows.append(window)
                        self.collected_labels.append(label)
                        self.collected_trial_ids.append(trial_id)
                        self.data_queue.put(('window_count', len(self.collected_windows)))
            else:
                # Check for data timeout
                if current_time - last_data_time > 3.0:
                    if not timeout_warning_sent:
                        self.data_queue.put(('warning', 'No data received - check ESP32 connection'))
                        self.data_queue.put(('connection_status', ('orange', 'No data')))
                        timeout_warning_sent = True

        # Collection complete
        self.data_queue.put(('done', None))

    def update_collection_ui(self):
        """Update UI from collection thread data."""
        needs_redraw = False

        try:
            # Process up to 10 messages per update cycle to prevent backlog
            for _ in range(10):
                msg_type, data = self.data_queue.get_nowait()

                if msg_type == 'prompt_with_countdown':
                    gesture_name, time_remaining, next_gesture = data
                    countdown_int = int(np.ceil(time_remaining))

                    if gesture_name == "rest" and next_gesture:
                        # During rest, show upcoming gesture
                        next_display = next_gesture.upper().replace("_", " ")
                        color = get_gesture_color(next_gesture)
                        display_text = f"{next_display} in {countdown_int}"
                    else:
                        # During gesture, show current gesture (user is holding it)
                        gesture_display = gesture_name.upper().replace("_", " ")
                        color = get_gesture_color(gesture_name)
                        if countdown_int > 0:
                            display_text = f"{gesture_display}  {countdown_int}"
                        else:
                            display_text = gesture_display

                    self.prompt_label.configure(text=display_text, text_color=color)

                elif msg_type == 'progress':
                    self.progress_bar.set(data)
                    remaining = self.scheduler.schedule.total_duration * (1 - data)
                    self.countdown_label.configure(text=f"Total: {remaining:.1f}s remaining")

                elif msg_type == 'samples_batch':
                    # Update plot data with batch of samples
                    for sample in data:
                        for i, val in enumerate(sample):
                            self.plot_data[i] = np.roll(self.plot_data[i], -1)
                            self.plot_data[i][-1] = val

                    # Update plot lines once per batch
                    for i in range(len(self.plot_lines)):
                        self.plot_lines[i].set_ydata(self.plot_data[i])
                    needs_redraw = True

                elif msg_type == 'window_count':
                    self.window_count_label.configure(text=f"Windows: {data}")

                elif msg_type == 'error':
                    # Show error and stop collection
                    self.status_label.configure(text=f"Error: {data}", text_color="red")
                    self._update_connection_status("red", "Disconnected")
                    messagebox.showerror("Collection Error", data)
                    self.stop_collection()
                    return

                elif msg_type == 'warning':
                    # Show warning but continue
                    self.status_label.configure(text=f"Warning: {data}", text_color="orange")

                elif msg_type == 'connection_status':
                    # Update connection indicator
                    color, text = data
                    self._update_connection_status(color, text)

                elif msg_type == 'done':
                    self.stop_collection()
                    return

        except queue.Empty:
            pass

        # Only redraw once per update cycle
        if needs_redraw:
            self.canvas.draw_idle()

        if self.is_collecting:
            self.after(50, self.update_collection_ui)

    def save_session(self):
        """Save the collected session."""
        if not self.collected_windows:
            messagebox.showwarning("No Data", "No data to save!")
            return

        user_id = self.user_id_entry.get() or USER_ID
        gestures = [g for g, var in self.gesture_vars.items() if var.get()]

        storage = SessionStorage()
        session_id = storage.generate_session_id(user_id)

        metadata = SessionMetadata(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            sampling_rate=SAMPLING_RATE_HZ,
            window_size_ms=WINDOW_SIZE_MS,
            num_channels=NUM_CHANNELS,
            gestures=gestures,
            notes=""
        )

        # Get session start time for label alignment
        session_start_time = None
        if self.scheduler and self.scheduler.session_start_time:
            session_start_time = self.scheduler.session_start_time

        filepath = storage.save_session(
            windows=self.collected_windows,
            labels=self.collected_labels,
            metadata=metadata,
            trial_ids=self.collected_trial_ids if self.collected_trial_ids else None,
            raw_samples=self.collected_raw_samples if self.collected_raw_samples else None,
            session_start_time=session_start_time
        )

        # Check if alignment was performed
        alignment_msg = ""
        if session_start_time and self.collected_raw_samples:
            alignment_msg = "\n\nLabel alignment: enabled"

        messagebox.showinfo("Saved", f"Session saved!\n\nID: {session_id}\nWindows: {len(self.collected_windows)}{alignment_msg}")

        # Update sidebar
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp):
            app.sidebar.update_status()

        # Reset for next collection
        self.collected_windows = []
        self.collected_labels = []
        self.collected_raw_samples = []
        self.save_button.configure(state="disabled")
        self.status_label.configure(text="Ready to collect")
        self.window_count_label.configure(text="Windows: 0")
        self.progress_bar.set(0)
        self.prompt_label.configure(text="READY", text_color="gray")

    def _refresh_ports(self):
        """Scan and populate available serial ports."""
        ports = serial.tools.list_ports.comports()
        port_names = ["Auto-detect"] + [p.device for p in ports]

        # Update dropdown values
        self.port_dropdown.configure(values=port_names)

        # Show port info
        if ports:
            self._update_connection_status("orange", f"Found {len(ports)} port(s)")
        else:
            self._update_connection_status("red", "No ports found")

    def _get_serial_port(self):
        """Get selected port, or None for auto-detect."""
        port = self.port_var.get()
        return None if port == "Auto-detect" else port

    def _update_connection_status(self, color: str, text: str):
        """Update the connection status indicator."""
        self.connection_status.configure(text=f"● {text}", text_color=color)

    def _toggle_connection(self):
        """Connect or disconnect from ESP32."""
        if self.is_connected:
            self._disconnect_device()
        else:
            self._connect_device()

    def _connect_device(self):
        """Connect to ESP32 with handshake."""
        print("\n" + "="*80)
        print("[DEBUG] _connect_device() called")
        port = self._get_serial_port()
        print(f"[DEBUG] Port: {port}")

        try:
            # Update UI to show connecting
            self._update_connection_status("orange", "Connecting...")
            self.connect_button.configure(state="disabled")
            self.update()  # Force UI update
            print("[DEBUG] UI updated - showing 'Connecting...'")

            # Create stream and connect
            self.stream = RealSerialStream(port=port)
            print("[DEBUG] Created RealSerialStream")
            device_info = self.stream.connect(timeout=5.0)
            print(f"[DEBUG] Connection successful: {device_info}")

            # Success!
            self.is_connected = True
            print("[DEBUG] Set is_connected = True")
            self._update_connection_status("green", f"Connected ({device_info.get('device', 'ESP32')})")
            self.connect_button.configure(text="Disconnect", state="normal")
            self.start_button.configure(state="normal")
            print("[DEBUG] Start button ENABLED")
            print(f"[DEBUG] Stream state: {self.stream.state}")
            print("="*80 + "\n")

        except TimeoutError as e:
            messagebox.showerror(
                "Connection Timeout",
                f"Device did not respond within 5 seconds.\n\n"
                f"Check that:\n"
                f"• ESP32 is powered on\n"
                f"• Correct firmware is flashed\n"
                f"• USB cable is properly connected"
            )
            self._update_connection_status("red", "Timeout")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

        except Exception as e:
            error_msg = f"Failed to connect:\n{e}"
            if "Permission denied" in str(e) or "Resource busy" in str(e):
                error_msg += "\n\nThe port may still be in use. Wait a few seconds and try again."
            elif "FileNotFoundError" in str(type(e).__name__):
                error_msg += f"\n\nPort not found. Try refreshing the port list."

            messagebox.showerror("Connection Error", error_msg)
            self._update_connection_status("red", "Failed")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

    def _disconnect_device(self):
        """Disconnect from ESP32."""
        try:
            if self.stream:
                self.stream.disconnect()
                # Give OS time to release the port
                time.sleep(0.5)

            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")
            self.start_button.configure(state="disabled")

        except Exception as e:
            messagebox.showwarning("Disconnect Warning", f"Error during disconnect: {e}")
            # Still mark as disconnected even if there was an error
            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")
            self.start_button.configure(state="disabled")

    def on_hide(self):
        """Stop collection when leaving page."""
        if self.is_collecting:
            self.stop_collection()

    def stop(self):
        """Stop everything."""
        self.is_collecting = False
        if self.stream:
            self.stream.stop()


# =============================================================================
# INSPECT SESSIONS PAGE
# =============================================================================

class InspectPage(BasePage):
    """Page for inspecting saved sessions with scrollable signal + label view."""

    # How many samples to show in the visible window at once
    VIEW_SAMPLES = 3000  # ~3 seconds at 1 kHz

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Inspect Sessions",
            "Browse session data — scroll through signals with gesture labels"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=0, minsize=220)
        self.content.grid_columnconfigure(1, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # ── Left panel ── Session list
        self.list_panel = ctk.CTkFrame(self.content, width=220)
        self.list_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.list_panel.grid_propagate(False)

        ctk.CTkLabel(self.list_panel, text="Sessions",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.session_listbox = ctk.CTkScrollableFrame(self.list_panel)
        self.session_listbox.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self.refresh_button = ctk.CTkButton(self.list_panel, text="Refresh",
                                            command=self.load_sessions)
        self.refresh_button.pack(pady=10)

        # ── Right panel ── Details + plot + slider
        self.details_panel = ctk.CTkFrame(self.content)
        self.details_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.details_panel.grid_columnconfigure(0, weight=1)
        self.details_panel.grid_rowconfigure(1, weight=1)  # plot row expands

        self.details_label = ctk.CTkLabel(
            self.details_panel,
            text="Select a session to view details",
            font=ctk.CTkFont(size=14),
            justify="left", anchor="w"
        )
        self.details_label.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))

        # Plot area (filled on session select, dark bg to avoid white flash)
        self.plot_frame = ctk.CTkFrame(self.details_panel, fg_color="#2b2b2b")
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Slider + zoom row
        self.controls_frame = ctk.CTkFrame(self.details_panel, fg_color="transparent")
        self.controls_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.controls_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.controls_frame, text="Position:",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 8))

        self.pos_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=1,
                                        command=self._on_slider)
        self.pos_slider.grid(row=0, column=1, sticky="ew")
        self.pos_slider.set(0)

        self.pos_label = ctk.CTkLabel(self.controls_frame, text="0.0 s",
                                      font=ctk.CTkFont(size=12), width=80)
        self.pos_label.grid(row=0, column=2, padx=(8, 0))

        # Zoom buttons
        zoom_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        zoom_frame.grid(row=0, column=3, padx=(16, 0))
        ctk.CTkButton(zoom_frame, text="−", width=32, command=self._zoom_out).pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="+", width=32, command=self._zoom_in).pack(side="left", padx=2)

        # Matplotlib objects
        self.fig = None
        self.canvas = None
        self.axes = []
        self.session_buttons = []

        # Loaded session state
        self._signal = None        # (total_samples, n_channels) continuous signal
        self._labels_per_sample = None  # label string per sample
        self._label_names = []
        self._n_channels = 0
        self._total_samples = 0
        self._view_start = 0       # current scroll position in samples
        self._view_len = self.VIEW_SAMPLES
        self._slider_debounce_id = None  # for debouncing slider updates

    # ── lifecycle ──

    def on_show(self):
        self.load_sessions()

    # ── session list ──

    def load_sessions(self):
        for btn in self.session_buttons:
            btn.destroy()
        self.session_buttons = []

        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            lbl = ctk.CTkLabel(self.session_listbox, text="No sessions found")
            lbl.pack(pady=10)
            self.session_buttons.append(lbl)
            return

        for session_id in sessions:
            info = storage.get_session_info(session_id)
            gestures = info['gestures']
            btn_text = f"{session_id}\n{info['num_windows']} win · {len(gestures)} gestures"

            btn = ctk.CTkButton(
                self.session_listbox,
                text=btn_text,
                font=ctk.CTkFont(size=11),
                height=55, anchor="w",
                command=lambda s=session_id: self.show_session(s)
            )
            btn.pack(fill="x", pady=3)
            self.session_buttons.append(btn)

    # ── load & show session ──

    def show_session(self, session_id: str):
        storage = SessionStorage()

        try:
            # Load raw windowed data WITHOUT transition filtering so we see
            # every window exactly as collected, labels included.
            X, y, label_names = storage.load_for_training(
                session_id, filter_transitions=False
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {e}")
            return

        n_windows, samples_per_window, n_channels = X.shape

        # Build a continuous signal by concatenating windows using hop-based
        # reconstruction. With 150-sample windows and 25-sample hop, consecutive
        # windows overlap by 125 samples. We take only the first `hop` samples
        # from each window (except the last, where we take the full window) to
        # avoid duplicated overlap regions.
        hop = HOP_SIZE_MS  # = 25 samples at 1 kHz (hop_size_ms == hop samples)
        total_samples = (n_windows - 1) * hop + samples_per_window

        signal = np.zeros((total_samples, n_channels), dtype=np.float32)
        labels_per_sample = np.empty(total_samples, dtype=object)
        labels_per_sample[:] = ""

        for i in range(n_windows):
            start = i * hop
            end = start + samples_per_window
            signal[start:end] = X[i]
            # Label this window's hop region (non-overlapping part)
            hop_end = start + hop if i < n_windows - 1 else end
            labels_per_sample[start:hop_end] = label_names[y[i]]

        # Fill any remaining gaps from the last window's tail
        mask = labels_per_sample == ""
        if mask.any():
            labels_per_sample[mask] = label_names[y[-1]]

        # Pre-compute centered signals (global mean removal) for smooth scrolling.
        # Using global mean ensures the signal doesn't jump when scrolling.
        centered = signal.astype(np.float64)
        for ch in range(n_channels):
            centered[:, ch] -= centered[:, ch].mean()

        # Store for scrolling
        self._signal = signal
        self._centered = centered
        self._labels_per_sample = labels_per_sample
        self._label_names = label_names
        self._n_channels = n_channels
        self._total_samples = total_samples
        self._view_start = 0

        # Update info text
        info = storage.get_session_info(session_id)
        duration_sec = total_samples / SAMPLING_RATE_HZ
        label_counts = {ln: int(np.sum(y == i)) for i, ln in enumerate(label_names)}
        counts_str = ", ".join(f"{n}: {c}" for n, c in sorted(label_counts.items()))
        info_text = (
            f"Session: {session_id}   |   "
            f"{n_windows} windows · {total_samples} samples · "
            f"{duration_sec:.1f} s · {n_channels} ch\n"
            f"Labels: {counts_str}"
        )
        self.details_label.configure(text=info_text)

        # Configure slider range
        max_start = max(0, total_samples - self._view_len)
        self.pos_slider.configure(to=max(max_start, 1))
        self.pos_slider.set(0)
        self._update_pos_label()

        # Build full-session plot (scroll with xlim, not rebuild)
        self._build_plot()

    # ── plotting ──

    def _build_plot(self):
        """Build plot skeleton once. Line data is updated via set_data() on scroll."""
        # Tear down old canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig:
            plt.close(self.fig)
        self.axes = []
        self._lines = []

        n_ch = min(self._n_channels, 4)
        self.fig = Figure(figsize=(12, max(2.5 * n_ch, 5)), dpi=100,
                          facecolor='#2b2b2b')

        duration_sec = self._total_samples / SAMPLING_RATE_HZ

        # Pre-build label colour strip as a tiny RGBA image (1 row, ~2k cols).
        # This replaces hundreds of axvspan patches with a single imshow per axis,
        # cutting per-frame render cost dramatically.
        from matplotlib.colors import to_rgba
        hop_ds = max(1, self._total_samples // 2000)  # downsample to ~2k pixels
        n_px = (self._total_samples + hop_ds - 1) // hop_ds
        label_img = np.zeros((1, n_px, 4), dtype=np.float32)
        for i in range(n_px):
            lbl = self._labels_per_sample[min(i * hop_ds, self._total_samples - 1)]
            label_img[0, i] = to_rgba(get_gesture_color(lbl), alpha=0.25)

        for ch in range(n_ch):
            ax = self.fig.add_subplot(n_ch, 1, ch + 1)
            ax.set_facecolor('#1e1e1e')
            self.axes.append(ax)

            # Fix y-axis to full signal range so it doesn't jump on scroll
            ch_min = float(self._centered[:, ch].min())
            ch_max = float(self._centered[:, ch].max())
            margin = (ch_max - ch_min) * 0.05
            ylo, yhi = ch_min - margin, ch_max + margin
            ax.set_ylim(ylo, yhi)

            # Label colour strip as a single imshow (replaces ~100 axvspan patches)
            ax.imshow(label_img, aspect='auto',
                      extent=[0, duration_sec, ylo, yhi],
                      origin='lower', zorder=1, interpolation='nearest')

            # Create empty line — data filled by _fill_view_data()
            line, = ax.plot([], [], color='#00ff88', linewidth=0.6, zorder=3)
            self._lines.append(line)

            ax.set_ylabel(f'Ch {ch}', color='white', fontsize=10, labelpad=10)
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.15, color='white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

            if ch < n_ch - 1:
                ax.tick_params(labelbottom=False)

        # X label on bottom axis
        if self.axes:
            self.axes[-1].set_xlabel('Time (s)', color='white', fontsize=10)

        # Legend at top
        if self.axes:
            from matplotlib.patches import Patch
            patches = [Patch(facecolor=get_gesture_color(n), alpha=0.35, label=n)
                       for n in self._label_names]
            self.axes[0].legend(handles=patches, loc='upper right', fontsize=8,
                                ncol=len(patches), framealpha=0.5,
                                facecolor='#333333', edgecolor='#555555',
                                labelcolor='white')

        self.fig.tight_layout(pad=1.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg='#2b2b2b', highlightthickness=0)
        widget.pack(fill="both", expand=True)

        # Fill initial view data and render
        self._fill_view_data()
        self.canvas.draw()

    def _fill_view_data(self):
        """Update line artists with only the visible window's data (~3k points)."""
        s = self._view_start
        e = min(s + self._view_len, self._total_samples)
        time_slice = np.arange(s, e) / SAMPLING_RATE_HZ

        for ch, line in enumerate(self._lines):
            line.set_data(time_slice, self._centered[s:e, ch])

        t_start = s / SAMPLING_RATE_HZ
        t_end = e / SAMPLING_RATE_HZ
        for ax in self.axes:
            ax.set_xlim(t_start, t_end)

    # ── slider / zoom callbacks ──

    def _on_slider(self, value):
        new_start = int(float(value))
        if new_start == self._view_start:
            return  # No change, skip redraw
        self._view_start = new_start
        self._update_pos_label()
        # Debounce: cancel pending draw and schedule a new one
        if self._slider_debounce_id is not None:
            self.after_cancel(self._slider_debounce_id)
        self._slider_debounce_id = self.after(8, self._scroll_draw)

    def _scroll_draw(self):
        """Fast redraw: update line data (~3k points) + xlim, no plot rebuild."""
        self._slider_debounce_id = None
        if self.canvas and self._lines:
            self._fill_view_data()
            self.canvas.draw()

    def _update_pos_label(self):
        t = self._view_start / SAMPLING_RATE_HZ
        self.pos_label.configure(text=f"{t:.1f} s")

    def _zoom_in(self):
        """Show fewer samples (zoom in)."""
        self._view_len = max(500, self._view_len // 2)
        self._clamp_view()
        self._scroll_draw()

    def _zoom_out(self):
        """Show more samples (zoom out)."""
        self._view_len = min(self._total_samples, self._view_len * 2)
        self._clamp_view()
        self._scroll_draw()

    def _clamp_view(self):
        max_start = max(0, self._total_samples - self._view_len)
        self._view_start = min(self._view_start, max_start)
        self.pos_slider.configure(to=max(max_start, 1))
        self.pos_slider.set(self._view_start)
        self._update_pos_label()


# =============================================================================
# TRAINING PAGE
# =============================================================================

class TrainingPage(BasePage):
    """Page for training the classifier."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Train Classifier",
            "Train LDA model on all collected sessions"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)

        # Sessions info
        self.info_frame = ctk.CTkFrame(self.content)
        self.info_frame.pack(fill="x", padx=20, pady=20)

        self.sessions_label = ctk.CTkLabel(
            self.info_frame,
            text="Loading sessions...",
            font=ctk.CTkFont(size=14)
        )
        self.sessions_label.pack(pady=10)

        # Model name input
        name_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        name_frame.pack(fill="x", padx=20, pady=(10, 0))

        ctk.CTkLabel(name_frame, text="Model name:", font=ctk.CTkFont(size=14)).pack(side="left")

        self.model_name_var = ctk.StringVar(value="emg_lda_classifier")
        self.model_name_entry = ctk.CTkEntry(
            name_frame, textvariable=self.model_name_var,
            width=250, placeholder_text="emg_lda_classifier"
        )
        self.model_name_entry.pack(side="left", padx=(10, 5))

        ctk.CTkLabel(
            name_frame, text=".joblib",
            font=ctk.CTkFont(size=14), text_color="gray"
        ).pack(side="left")

        # Model type selector
        type_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        type_frame.pack(fill="x", padx=20, pady=(10, 0))

        ctk.CTkLabel(type_frame, text="Model type:", font=ctk.CTkFont(size=14)).pack(side="left")

        self.model_type_var = ctk.StringVar(value="LDA")
        self.model_type_selector = ctk.CTkSegmentedButton(
            type_frame, values=["LDA", "QDA"],
            variable=self.model_type_var,
        )
        self.model_type_selector.pack(side="left", padx=(10, 10))

        self.model_type_desc = ctk.CTkLabel(
            type_frame,
            text="Linear — fast, exportable to ESP32",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.model_type_desc.pack(side="left")

        self.model_type_var.trace_add("write", self._on_model_type_changed)

        # QDA regularisation slider (only active when QDA is selected)
        reg_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        reg_frame.pack(fill="x", padx=20, pady=(6, 0))

        ctk.CTkLabel(reg_frame, text="reg_param:", font=ctk.CTkFont(size=14)).pack(side="left")

        self.reg_param_var = ctk.DoubleVar(value=0.1)
        self.reg_param_slider = ctk.CTkSlider(
            reg_frame, from_=0.0, to=1.0, variable=self.reg_param_var,
            width=180, state="disabled",
            command=lambda v: self.reg_param_label.configure(text=f"{v:.2f}"),
        )
        self.reg_param_slider.pack(side="left", padx=(10, 6))

        self.reg_param_label = ctk.CTkLabel(
            reg_frame, text="0.10", font=ctk.CTkFont(size=13), width=40
        )
        self.reg_param_label.pack(side="left")

        self.reg_param_desc = ctk.CTkLabel(
            reg_frame, text="(enable QDA to adjust — 0=flexible, 1=LDA-like)",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.reg_param_desc.pack(side="left", padx=(8, 0))

        # Train button
        self.train_button = ctk.CTkButton(
            self.content,
            text="Train on All Sessions",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            command=self.train_model
        )
        self.train_button.pack(pady=20)

        # Export button
        self.export_button = ctk.CTkButton(
            self.content,
            text="Export for ESP32",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="green",
            state="disabled",
            command=self.export_model
        )
        self.export_button.pack(pady=5)

        # Advanced training (ensemble + MLP)
        adv_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        adv_frame.pack(fill="x", padx=20, pady=(15, 0))

        ctk.CTkLabel(
            adv_frame, text="Advanced (ESP32 only):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left")

        self.train_ensemble_button = ctk.CTkButton(
            adv_frame, text="Train Ensemble",
            font=ctk.CTkFont(size=13), height=34,
            fg_color="#8B5CF6", hover_color="#7C3AED",
            state="disabled",
            command=self._train_ensemble
        )
        self.train_ensemble_button.pack(side="left", padx=(10, 5))

        self.train_mlp_button = ctk.CTkButton(
            adv_frame, text="Train MLP",
            font=ctk.CTkFont(size=13), height=34,
            fg_color="#8B5CF6", hover_color="#7C3AED",
            state="disabled",
            command=self._train_mlp
        )
        self.train_mlp_button.pack(side="left", padx=5)

        self.adv_desc = ctk.CTkLabel(
            adv_frame,
            text="(train base LDA first)",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.adv_desc.pack(side="left", padx=(8, 0))

        # Progress
        self.progress_bar = ctk.CTkProgressBar(self.content, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        # Results
        self.results_frame = ctk.CTkFrame(self.content)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.results_text = ctk.CTkTextbox(self.results_frame, font=ctk.CTkFont(family="Courier", size=12))
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.classifier = None

    def on_show(self):
        """Update session info when shown."""
        self.update_session_info()

    def update_session_info(self):
        """Update the sessions information display."""
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            self.sessions_label.configure(text="No sessions found. Collect data first!")
            self.train_button.configure(state="disabled")
            return

        total_windows = 0
        info_lines = [f"Found {len(sessions)} session(s):\n"]

        for session_id in sessions:
            info = storage.get_session_info(session_id)
            info_lines.append(f"  - {session_id}: {info['num_windows']} windows")
            total_windows += info['num_windows']

        info_lines.append(f"\nTotal: {total_windows} windows")

        self.sessions_label.configure(text="\n".join(info_lines))
        self.train_button.configure(state="normal")

    def _get_model_path(self) -> Path:
        """Build model save path from the user-entered name."""
        name = self.model_name_var.get().strip()
        if not name:
            name = "emg_lda_classifier"
        # Sanitize: remove extension if user typed one, strip unsafe chars
        name = name.replace(".joblib", "").replace("/", "_").replace("\\", "_")
        return MODEL_DIR / f"{name}.joblib"

    def _on_model_type_changed(self, *args):
        """Update description, model name, and reg_param slider when model type changes."""
        mt = self.model_type_var.get()
        if mt == "QDA":
            self.model_type_desc.configure(text="Quadratic — flexible boundaries, laptop-only")
            self.reg_param_slider.configure(state="normal")
            self.reg_param_desc.configure(text="0=flexible quadratic, 1=LDA-like", text_color="white")
            # Auto-suggest a QDA filename if still on the default LDA name
            if self.model_name_var.get().strip() in ("", "emg_lda_classifier"):
                self.model_name_var.set("emg_qda_classifier")
        else:
            self.model_type_desc.configure(text="Linear — fast, exportable to ESP32")
            self.reg_param_slider.configure(state="disabled")
            self.reg_param_desc.configure(
                text="(enable QDA to adjust — 0=flexible, 1=LDA-like)", text_color="gray"
            )
            if self.model_name_var.get().strip() in ("", "emg_qda_classifier"):
                self.model_name_var.set("emg_lda_classifier")

    def train_model(self):
        """Train the model on all sessions."""
        self.train_button.configure(state="disabled")
        self.results_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.status_label.configure(text="Loading data...")

        # Capture model path, type, and reg_param on UI thread (StringVar isn't thread-safe)
        model_save_path = self._get_model_path()
        model_type = self.model_type_var.get().lower()
        reg_param = float(self.reg_param_var.get())

        # Run in thread to not block UI
        thread = threading.Thread(
            target=self._train_thread, args=(model_save_path, model_type, reg_param), daemon=True
        )
        thread.start()

    def _train_thread(self, model_save_path: Path, model_type: str = "lda", reg_param: float = 0.1):
        """Training thread."""
        try:
            storage = SessionStorage()

            # Load data
            self.after(0, lambda: self.status_label.configure(text="Loading all sessions..."))
            self.after(0, lambda: self.progress_bar.set(0.2))

            X, y, trial_ids, session_indices, label_names, loaded_sessions = storage.load_all_for_training()

            n_trials = len(np.unique(trial_ids))
            n_sessions = len(np.unique(session_indices))
            self.after(0, lambda: self._log(f"Loaded {X.shape[0]} windows from {len(loaded_sessions)} sessions"))
            self.after(0, lambda: self._log(f"Unique trials: {n_trials} (for proper train/test splitting)"))
            self.after(0, lambda ns=n_sessions: self._log(f"Session normalization: {ns} sessions will be z-scored independently"))
            self.after(0, lambda: self._log(f"Labels: {label_names}\n"))

            # Train
            self.after(0, lambda mt=model_type: self.status_label.configure(text=f"Training {mt.upper()} classifier..."))
            self.after(0, lambda: self.progress_bar.set(0.5))

            self.classifier = EMGClassifier(model_type=model_type, reg_param=reg_param)
            self.classifier.train(X, y, label_names, session_indices=session_indices)

            self.after(0, lambda: self._log("Training complete!\n"))

            # Cross-validation (trial-level to prevent leakage)
            self.after(0, lambda: self.status_label.configure(text="Running cross-validation (trial-level)..."))
            self.after(0, lambda: self.progress_bar.set(0.7))

            cv_scores = self.classifier.cross_validate(X, y, trial_ids=trial_ids, cv=5, session_indices=session_indices)

            self.after(0, lambda: self._log(f"Cross-validation scores: {cv_scores.round(3)}"))
            self.after(0, lambda: self._log(f"Mean accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)\n"))

            # Feature importance
            self.after(0, lambda: self._log("Feature importance (top 8):"))
            importance = self.classifier.get_feature_importance()
            for i, (name, score) in enumerate(list(importance.items())[:8]):
                self.after(0, lambda n=name, s=score: self._log(f"  {n}: {s:.3f}"))

            # Save model
            self.after(0, lambda: self.status_label.configure(text="Saving model..."))
            self.after(0, lambda: self.progress_bar.set(0.9))

            model_path = self.classifier.save(model_save_path)

            self.after(0, lambda: self._log(f"\nModel saved to: {model_path}"))
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.status_label.configure(text="Training complete!"))

            # Update sidebar
            self.after(0, lambda: self._update_sidebar())

        except Exception as e:
            self.after(0, lambda: self._log(f"\nError: {e}"))
            self.after(0, lambda: self.status_label.configure(text="Training failed!"))

        finally:
            self.after(0, lambda: self.train_button.configure(state="normal"))
            # Only enable export if an LDA model was trained (QDA can't export to C)
            can_export = self.classifier and self.classifier.model_type == "lda"
            self.after(0, lambda: self.export_button.configure(
                state="normal" if can_export else "disabled"
            ))
            # Enable advanced training buttons after successful LDA training
            if can_export:
                self.after(0, lambda: self.train_ensemble_button.configure(state="normal"))
                self.after(0, lambda: self.train_mlp_button.configure(state="normal"))
                self.after(0, lambda: self.adv_desc.configure(
                    text="Ensemble: 3-specialist LDA stacker  |  MLP: int8 neural net"
                ))

    def _train_ensemble(self):
        """Train the 3-specialist + meta-LDA ensemble (runs train_ensemble.py)."""
        self.train_ensemble_button.configure(state="disabled")
        self._log("\n--- Training Ensemble ---")
        self.status_label.configure(text="Training ensemble (3 specialist LDAs + meta-LDA)...")
        self.progress_bar.set(0.3)

        def _run():
            try:
                script = str(Path(__file__).parent / "train_ensemble.py")
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True, text=True, timeout=300
                )
                output = result.stdout + result.stderr
                self.after(0, lambda: self._log(output))
                if result.returncode == 0:
                    self.after(0, lambda: self._log("\nEnsemble training complete!"))
                    self.after(0, lambda: self.status_label.configure(text="Ensemble trained!"))
                else:
                    self.after(0, lambda: self._log(f"\nEnsemble training failed (exit code {result.returncode})"))
                    self.after(0, lambda: self.status_label.configure(text="Ensemble training failed"))
            except Exception as e:
                self.after(0, lambda: self._log(f"\nEnsemble error: {e}"))
                self.after(0, lambda: self.status_label.configure(text="Ensemble training failed"))
            finally:
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self.train_ensemble_button.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _train_mlp(self):
        """Train the int8 MLP model (runs train_mlp_tflite.py).

        TensorFlow requires Python <=3.12.  Try ``py -3.12`` first (Windows
        launcher), fall back to the current interpreter.
        """
        self.train_mlp_button.configure(state="disabled")
        self._log("\n--- Training MLP (TFLite int8) ---")
        self.status_label.configure(text="Training MLP neural network...")
        self.progress_bar.set(0.3)

        def _run():
            try:
                script = str(Path(__file__).parent / "train_mlp_tflite.py")
                # TensorFlow needs Python <=3.12; try py launcher first
                python_cmd = [sys.executable]
                try:
                    probe = subprocess.run(
                        ["py", "-3.12", "-c", "import tensorflow"],
                        capture_output=True, timeout=30,
                    )
                    if probe.returncode == 0:
                        python_cmd = ["py", "-3.12"]
                        self.after(0, lambda: self._log("Using Python 3.12 (TensorFlow compatible)"))
                except FileNotFoundError:
                    pass
                result = subprocess.run(
                    python_cmd + [script],
                    capture_output=True, text=True, timeout=600
                )
                output = result.stdout + result.stderr
                self.after(0, lambda: self._log(output))
                if result.returncode == 0:
                    self.after(0, lambda: self._log("\nMLP training complete!"))
                    self.after(0, lambda: self.status_label.configure(text="MLP trained!"))
                else:
                    self.after(0, lambda: self._log(f"\nMLP training failed (exit code {result.returncode})"))
                    self.after(0, lambda: self.status_label.configure(text="MLP training failed"))
            except Exception as e:
                self.after(0, lambda: self._log(f"\nMLP error: {e}"))
                self.after(0, lambda: self.status_label.configure(text="MLP training failed"))
            finally:
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self.train_mlp_button.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def export_model(self):
        """Export trained model to C header (LDA only)."""
        if not self.classifier or not self.classifier.is_trained:
            messagebox.showerror("Error", "No trained model to export!")
            return

        if self.classifier.model_type != "lda":
            messagebox.showerror(
                "Export Not Supported",
                "QDA models cannot be exported to C header.\n\n"
                "QDA uses per-class covariance matrices which don't reduce to\n"
                "simple weights/biases. Train an LDA model to export for ESP32."
            )
            return

        # Default path in ESP32 project
        default_path = Path("EMG_Arm/src/core/model_weights.h").absolute()
        
        # Ask user for location, defaulting to the ESP32 project source
        filename = tk.filedialog.asksaveasfilename(
            title="Export Model Header",
            initialdir=default_path.parent,
            initialfile=default_path.name,
            filetypes=[("C Header", "*.h")]
        )
        
        if filename:
            try:
                path = self.classifier.export_to_header(filename)
                self._log(f"\nExported model to: {path}")
                messagebox.showinfo("Export Success", f"Model exported to:\n{path}\n\nRecompile ESP32 firmware to apply.")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")

    def _log(self, text: str):
        """Add text to results."""
        self.results_text.insert("end", text + "\n")
        self.results_text.see("end")

    def _update_sidebar(self):
        """Safely update the sidebar."""
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp):
            app.sidebar.update_status()


# =============================================================================
# CALIBRATION PAGE
# =============================================================================

class CalibrationPage(BasePage):
    """
    Session calibration — aligns the current-session EMG feature distribution
    to the training distribution so the classifier works reliably across sessions.

    Workflow:
      1. Load a trained model (needs training stats stored during training).
      2. Connect to ESP32.
      3. Click "Start Calibration": hold each gesture for 5 seconds when prompted.
      4. Click "Apply Calibration": stores the fitted transform in the app so
         PredictionPage uses it automatically in Laptop inference mode.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Calibrate",
            "Align current session to training data — fixes electrode placement drift"
        )

        # Page state
        self.is_calibrating = False
        self.is_connected = False
        self.classifier = None
        self.stream = None
        self.calib_thread = None
        self._calib_gestures: list[str] = []   # Populated from model labels at start

        # Two-column layout
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self.content)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.right_panel = ctk.CTkFrame(self.content)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._setup_left_panel()
        self._setup_right_panel()

    # ------------------------------------------------------------------
    # Left panel — controls
    # ------------------------------------------------------------------

    def _setup_left_panel(self):
        p = self.left_panel

        # Model picker
        ctk.CTkLabel(p, text="Trained Model:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", padx=20, pady=(20, 0)
        )
        model_row = ctk.CTkFrame(p, fg_color="transparent")
        model_row.pack(fill="x", padx=20, pady=(5, 0))

        self.model_var = ctk.StringVar(value="No models found")
        self.model_dropdown = ctk.CTkOptionMenu(model_row, variable=self.model_var, width=240)
        self.model_dropdown.pack(side="left")

        self.refresh_models_btn = ctk.CTkButton(
            model_row, text="⟳", width=30, command=self._refresh_models
        )
        self.refresh_models_btn.pack(side="left", padx=(5, 0))

        self.load_model_btn = ctk.CTkButton(
            p, text="Load Model", height=34, command=self._load_model
        )
        self.load_model_btn.pack(fill="x", padx=20, pady=(8, 0))

        self.model_status_label = ctk.CTkLabel(
            p, text="No model loaded", font=ctk.CTkFont(size=12), text_color="orange"
        )
        self.model_status_label.pack(anchor="w", padx=20, pady=(4, 0))

        # Divider
        ctk.CTkFrame(p, height=1, fg_color="gray40").pack(fill="x", padx=20, pady=14)

        # ESP32 connection
        ctk.CTkLabel(p, text="ESP32 Connection:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", padx=20
        )
        port_row = ctk.CTkFrame(p, fg_color="transparent")
        port_row.pack(fill="x", padx=20, pady=(5, 0))

        ctk.CTkLabel(port_row, text="Port:").pack(side="left")
        self.port_var = ctk.StringVar(value="Auto-detect")
        self.port_dropdown = ctk.CTkOptionMenu(
            port_row, variable=self.port_var, values=["Auto-detect"], width=140
        )
        self.port_dropdown.pack(side="left", padx=(8, 4))

        self.refresh_ports_btn = ctk.CTkButton(
            port_row, text="⟳", width=30, command=self._refresh_ports
        )
        self.refresh_ports_btn.pack(side="left")

        conn_row = ctk.CTkFrame(p, fg_color="transparent")
        conn_row.pack(fill="x", padx=20, pady=(5, 0))

        self.connect_btn = ctk.CTkButton(
            conn_row, text="Connect", width=100, height=28, command=self._toggle_connection
        )
        self.connect_btn.pack(side="left", padx=(0, 10))

        self.conn_status_label = ctk.CTkLabel(
            conn_row, text="● Disconnected", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.conn_status_label.pack(side="left")

        # Divider
        ctk.CTkFrame(p, height=1, fg_color="gray40").pack(fill="x", padx=20, pady=14)

        # Action buttons
        self.start_btn = ctk.CTkButton(
            p,
            text="Start Calibration",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            state="disabled",
            command=self._start_calibration,
        )
        self.start_btn.pack(fill="x", padx=20, pady=(0, 8))

        self.apply_btn = ctk.CTkButton(
            p,
            text="Apply Calibration to Prediction",
            font=ctk.CTkFont(size=13),
            height=40,
            fg_color="#28a745",
            hover_color="#1e7e34",
            state="disabled",
            command=self._apply_calibration,
        )
        self.apply_btn.pack(fill="x", padx=20, pady=(0, 8))

        # Log box
        ctk.CTkLabel(p, text="Log:", font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=20, pady=(10, 0)
        )
        self.log_box = ctk.CTkTextbox(
            p, font=ctk.CTkFont(family="Courier", size=11), height=160
        )
        self.log_box.pack(fill="x", padx=20, pady=(4, 20))

        self._refresh_models()
        self._refresh_ports()

    # ------------------------------------------------------------------
    # Right panel — gesture display and countdown
    # ------------------------------------------------------------------

    def _setup_right_panel(self):
        p = self.right_panel

        # Overall progress
        ctk.CTkLabel(p, text="Overall progress:", font=ctk.CTkFont(size=13)).pack(
            pady=(20, 4)
        )
        self.overall_progress = ctk.CTkProgressBar(p, width=320)
        self.overall_progress.pack()
        self.overall_progress.set(0)

        self.progress_text = ctk.CTkLabel(
            p, text="0 / 0 gestures", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.progress_text.pack(pady=(4, 16))

        # Big gesture name
        self.gesture_label = ctk.CTkLabel(
            p,
            text="---",
            font=ctk.CTkFont(size=64, weight="bold"),
            text_color="gray",
        )
        self.gesture_label.pack(pady=(10, 6))

        # Instruction text
        self.instruction_label = ctk.CTkLabel(
            p,
            text="Load a model and connect to begin",
            font=ctk.CTkFont(size=15),
            text_color="gray",
        )
        self.instruction_label.pack(pady=4)

        # Countdown (remaining seconds in current gesture)
        self.countdown_label = ctk.CTkLabel(
            p,
            text="",
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color="#FFD700",
        )
        self.countdown_label.pack(pady=8)

        # Per-gesture progress bar
        ctk.CTkLabel(
            p, text="Current gesture:", font=ctk.CTkFont(size=12), text_color="gray"
        ).pack(pady=(8, 2))
        self.gesture_progress = ctk.CTkProgressBar(p, width=320)
        self.gesture_progress.pack()
        self.gesture_progress.set(0)

        # Applied status
        self.calib_applied_label = ctk.CTkLabel(
            p, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color="green"
        )
        self.calib_applied_label.pack(pady=16)

    # ------------------------------------------------------------------
    # on_show / on_hide
    # ------------------------------------------------------------------

    def on_show(self):
        self._refresh_models()
        # Reflect whether calibration is already applied
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp) and app.calibrated_classifier is not None:
            self.calib_applied_label.configure(
                text="Calibration active — go to Live Prediction to use it"
            )
        else:
            self.calib_applied_label.configure(text="")

    def on_hide(self):
        if self.is_calibrating:
            self.is_calibrating = False
            if self.stream:
                try:
                    self.stream.stop()
                except Exception:
                    pass

    def stop(self):
        self.is_calibrating = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _refresh_models(self):
        models = EMGClassifier.list_saved_models()
        if models:
            names = [p.name for p in models]
            self.model_dropdown.configure(values=names)
            latest = max(models, key=lambda p: p.stat().st_mtime)
            self.model_var.set(latest.name)
        else:
            self.model_dropdown.configure(values=["No models found"])
            self.model_var.set("No models found")

    def _get_model_path(self):
        name = self.model_var.get()
        if name == "No models found":
            return None
        path = MODEL_DIR / name
        return path if path.exists() else None

    def _load_model(self):
        path = self._get_model_path()
        if not path:
            messagebox.showerror("No Model", "Select a model from the dropdown first.")
            return
        try:
            self.classifier = EMGClassifier.load(path)
            if self.classifier.calibration_transform.has_training_stats:
                mt = self.classifier.model_type.upper()
                rp = (f", reg_param={self.classifier.reg_param:.2f}"
                      if self.classifier.model_type == "qda" else "")
                sn = getattr(self.classifier, 'session_normalized', False)
                sn_str = "" if sn else "  [!old — retrain recommended]"
                status_color = "green" if sn else "orange"
                self.model_status_label.configure(
                    text=f"Loaded: {path.name}  [{mt}{rp}]{sn_str}",
                    text_color=status_color,
                )
                self._log(f"Model loaded: {path.name}  [{mt}{rp}]")
                self._log(f"Gestures: {self.classifier.label_names}")
                if not sn:
                    self._log("WARNING: This model was trained without session normalization.")
                    self._log("  Calibration will work but may be less accurate, especially for QDA.")
                    self._log("  Retrain to get proper calibration support.")
            else:
                self.model_status_label.configure(
                    text=f"Loaded (old model — retrain to enable calibration)",
                    text_color="orange",
                )
                self._log("Warning: model has no training stats.")
                self._log("Retrain the model to enable calibration support.")
            self._update_start_button()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{e}")

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        port_names = ["Auto-detect"] + [p.device for p in ports]
        self.port_dropdown.configure(values=port_names)

    def _get_port(self):
        p = self.port_var.get()
        return None if p == "Auto-detect" else p

    def _toggle_connection(self):
        if self.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._get_port()
        try:
            self.conn_status_label.configure(text="● Connecting...", text_color="orange")
            self.connect_btn.configure(state="disabled")
            self.update()
            self.stream = RealSerialStream(port=port)
            device_info = self.stream.connect(timeout=5.0)
            self.is_connected = True
            self.conn_status_label.configure(
                text=f"● Connected ({device_info.get('device', 'ESP32')})",
                text_color="green",
            )
            self.connect_btn.configure(text="Disconnect", state="normal")
            self._log("ESP32 connected")
            self._update_start_button()
        except TimeoutError:
            self.conn_status_label.configure(text="● Timeout", text_color="red")
            self.connect_btn.configure(state="normal")
            messagebox.showerror("Timeout", "ESP32 did not respond within 5 seconds.")
            if self.stream:
                try:
                    self.stream.disconnect()
                except Exception:
                    pass
                self.stream = None
        except Exception as e:
            self.conn_status_label.configure(text="● Failed", text_color="red")
            self.connect_btn.configure(state="normal")
            messagebox.showerror("Connection Error", str(e))
            if self.stream:
                try:
                    self.stream.disconnect()
                except Exception:
                    pass
                self.stream = None

    def _disconnect(self):
        try:
            if self.stream:
                self.stream.disconnect()
                time.sleep(0.3)
        except Exception:
            pass
        self.is_connected = False
        self.stream = None
        self.conn_status_label.configure(text="● Disconnected", text_color="gray")
        self.connect_btn.configure(text="Connect", state="normal")
        self._update_start_button()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _update_start_button(self):
        can_start = (
            self.classifier is not None
            and self.classifier.calibration_transform.has_training_stats
            and self.is_connected
            and not self.is_calibrating
        )
        self.start_btn.configure(state="normal" if can_start else "disabled")

    def _log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    # ------------------------------------------------------------------
    # Calibration logic
    # ------------------------------------------------------------------

    def _start_calibration(self):
        if self.is_calibrating:
            return

        self.is_calibrating = True
        self.apply_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.calib_applied_label.configure(text="")
        self.overall_progress.set(0)
        self.gesture_progress.set(0)
        self._log("\n--- Starting calibration ---")
        self._log(f"Each gesture: {int(CALIB_PREP_SEC)}s prep → {int(CALIB_DURATION_SEC)}s hold")

        try:
            self.stream.start()
            self.stream.running = True
        except Exception as e:
            messagebox.showerror("Stream Error", f"Could not start EMG stream:\n{e}")
            self.is_calibrating = False
            self._update_start_button()
            return

        # Build gesture order: rest first, then others sorted
        labels = self.classifier.label_names
        gestures = ["rest"] + sorted(g for g in labels if g != "rest")
        self._calib_gestures = gestures
        self.progress_text.configure(text=f"0 / {len(gestures)} gestures")

        import threading as _threading
        self.calib_thread = _threading.Thread(
            target=self._calibration_thread, args=(gestures,), daemon=True
        )
        self.calib_thread.start()

    def _calibration_thread(self, gestures: list):
        """
        Background thread: walks through each gesture in two phases.

        Phase 1 — Preparation (CALIB_PREP_SEC seconds):
          Show the gesture name in yellow with a whole-second countdown so
          the user has time to form the gesture before recording begins.
          Serial samples are drained but discarded to keep the buffer fresh.

        Phase 2 — Collection (CALIB_DURATION_SEC seconds):
          Gesture name switches to its gesture colour.  EMG windows are
          extracted and stored.  A decimal countdown shows remaining time.

        All UI mutations go through self.after() for thread safety.
        """
        parser = EMGParser(num_channels=NUM_CHANNELS)
        windower = Windower(
            window_size_ms=WINDOW_SIZE_MS,
            sample_rate=SAMPLING_RATE_HZ,
            hop_size_ms=HOP_SIZE_MS,
        )
        all_features = []
        all_labels = []
        rms_by_gesture: dict[str, list[float]] = {}  # AC-RMS per window, keyed by gesture
        n_gestures = len(gestures)

        try:
            for g_idx, gesture in enumerate(gestures):
                if not self.is_calibrating:
                    return

                display_name = gesture.upper().replace("_", " ")
                gesture_color = get_gesture_color(gesture)

                # ── Phase 1: Preparation countdown ──────────────────────────
                # Show gesture in yellow so the user knows what's coming and
                # can start forming the gesture before recording begins.
                self.after(0, lambda t=display_name: self.gesture_label.configure(
                    text=t, text_color="#FFD700"
                ))
                self.after(0, lambda: self.instruction_label.configure(
                    text="Get ready..."
                ))
                self.after(0, lambda: self.gesture_progress.set(0))
                self.after(0, lambda: self.countdown_label.configure(
                    text=str(int(CALIB_PREP_SEC))
                ))

                prep_start = time.perf_counter()
                last_ui_time = prep_start

                while self.is_calibrating:
                    elapsed = time.perf_counter() - prep_start
                    if elapsed >= CALIB_PREP_SEC:
                        break

                    now = time.perf_counter()
                    if now - last_ui_time >= 0.05:
                        remaining = CALIB_PREP_SEC - elapsed
                        # Show whole-second countdown: 3 → 2 → 1
                        tick = max(1, int(np.ceil(remaining)))
                        self.after(0, lambda s=tick: self.countdown_label.configure(
                            text=str(s)
                        ))
                        last_ui_time = now

                    # Drain serial buffer — keeps it fresh for collection
                    self.stream.readline()

                if not self.is_calibrating:
                    return

                # Brief "GO!" flash before collection starts
                self.after(0, lambda: self.countdown_label.configure(text="GO!"))
                time.sleep(0.2)

                # ── Phase 2: Collection ─────────────────────────────────────
                # Switch to gesture colour — this signals "recording now"
                self.after(0, lambda t=display_name, c=gesture_color: (
                    self.gesture_label.configure(text=t, text_color=c)
                ))
                self.after(0, lambda d=int(CALIB_DURATION_SEC): self.instruction_label.configure(
                    text=f"Hold this gesture for {d} seconds"
                ))

                gesture_start = time.perf_counter()
                windows_collected = 0
                last_ui_time = gesture_start

                while self.is_calibrating:
                    elapsed = time.perf_counter() - gesture_start
                    if elapsed >= CALIB_DURATION_SEC:
                        break

                    now = time.perf_counter()
                    if now - last_ui_time >= 0.05:
                        remaining = CALIB_DURATION_SEC - elapsed
                        progress = elapsed / CALIB_DURATION_SEC
                        self.after(0, lambda r=remaining: self.countdown_label.configure(
                            text=f"{r:.1f}s"
                        ))
                        self.after(0, lambda p=progress: self.gesture_progress.set(p))
                        last_ui_time = now

                    line = self.stream.readline()
                    if not line:
                        continue

                    sample = parser.parse_line(line)
                    if sample is None:
                        continue

                    window = windower.add_sample(sample)
                    if window is not None:
                        w_np = window.to_numpy()
                        feat = self.classifier.feature_extractor.extract_features_window(w_np)
                        all_features.append(feat)
                        all_labels.append(gesture)
                        windows_collected += 1
                        w_ac = w_np - w_np.mean(axis=0)  # remove per-window DC offset
                        ac_rms = float(np.sqrt(np.mean(w_ac ** 2)))
                        rms_by_gesture.setdefault(gesture, []).append(ac_rms)

                # Log and advance overall progress bar
                overall_prog = (g_idx + 1) / n_gestures
                self.after(0, lambda g=gesture, w=windows_collected, p=overall_prog, i=g_idx, n=n_gestures: (
                    self._log(f"  {g}: {w} windows"),
                    self.overall_progress.set(p),
                    self.progress_text.configure(text=f"{i + 1} / {n} gestures"),
                ))

        finally:
            self.stream.stop()

        if not self.is_calibrating:
            # User navigated away — abort
            return

        self.is_calibrating = False

        if not all_features:
            self.after(0, lambda: messagebox.showerror(
                "No Data", "No windows were collected. Check the EMG connection."
            ))
            self.after(0, self._update_start_button)
            return

        # Fit the calibration transform
        X_calib = np.array(all_features)
        try:
            self.classifier.calibration_transform.fit_from_calibration(X_calib, all_labels)

            # Set rest energy gate from raw window RMS (must be done here, not in
            # fit_from_calibration, because extracted features are amplitude-normalized).
            #
            # Scan every candidate threshold and pick the one that minimises:
            #   rest_miss_rate   (rest windows above gate → reach LDA → may jitter)
            #   gesture_miss_rate (gesture windows below gate → blocked → feel hard)
            # Equal weighting by default; prints the full breakdown so you can see
            # whether the two distributions actually separate cleanly.
            if "rest" in rms_by_gesture:
                rest_arr   = np.array(rms_by_gesture["rest"])
                active_arr = np.concatenate([
                    np.array(v) for g, v in rms_by_gesture.items() if g != "rest"
                ])

                # Print distribution summary for diagnosis
                self.after(0, lambda: self._log("\nRMS energy distribution (AC, pre-gate):"))
                self.after(0, lambda r=rest_arr: self._log(
                    f"  rest   — p50={np.percentile(r,50):.1f}  p95={np.percentile(r,95):.1f}  max={r.max():.1f}"))
                for g, v in rms_by_gesture.items():
                    if g == "rest":
                        continue
                    va = np.array(v)
                    self.after(0, lambda g=g, va=va: self._log(
                        f"  {g:<12s}— p5={np.percentile(va,5):.1f}  p50={np.percentile(va,50):.1f}  min={va.min():.1f}"))

                # Scan candidates from rest min to active max
                candidates = np.linspace(rest_arr.min(), active_arr.max(), 1000)
                best_t, best_err = float(rest_arr.max()), float("inf")
                for t in candidates:
                    rest_miss    = float((rest_arr   > t).mean())   # rest slips to LDA
                    gesture_miss = float((active_arr <= t).mean())   # gesture blocked
                    err = rest_miss + gesture_miss
                    if err < best_err:
                        best_err, best_t = err, float(t)

                rest_miss_at_best    = float((rest_arr   > best_t).mean()) * 100
                gesture_miss_at_best = float((active_arr <= best_t).mean()) * 100

                self.classifier.calibration_transform.rest_energy_threshold = best_t
                print(f"[Calibration] Optimal rest gate: {best_t:.2f}  "
                      f"(rest_miss={rest_miss_at_best:.1f}%, gesture_miss={gesture_miss_at_best:.1f}%)")
                self.after(0, lambda t=best_t, rm=rest_miss_at_best, gm=gesture_miss_at_best: (
                    self._log(f"\nOptimal rest gate: {t:.2f}"),
                    self._log(f"  rest above gate (may jitter): {rm:.1f}%"),
                    self._log(f"  gestures below gate (feel hard): {gm:.1f}%"),
                ))

            # Warn when rest energy overlaps any gesture — indicates bad electrode contact
            if "rest" in rms_by_gesture:
                for g, v in rms_by_gesture.items():
                    if g != "rest" and np.array(v).min() < rest_arr.max():
                        self.after(0, lambda g=g: self._log(
                            f"\nWARNING: rest energy overlaps {g}. "
                            f"Electrode placement may be poor — adjust and recalibrate."))

            self.after(0, self._on_calibration_complete)
        except Exception as e:
            self.after(0, lambda err=e: messagebox.showerror(
                "Calibration Error", f"Failed to fit transform:\n{err}"
            ))

        self.after(0, self._update_start_button)

    def _on_calibration_complete(self):
        """Called on the main thread when calibration data collection finishes."""
        self.gesture_label.configure(text="DONE!", text_color="#28a745")
        self.instruction_label.configure(
            text="Calibration collected. Click 'Apply' to activate."
        )
        self.countdown_label.configure(text="")
        self.gesture_progress.set(1.0)
        self.apply_btn.configure(state="normal")

        # Show z-score normalization diagnostics so the user can spot bad calibration
        ct = self.classifier.calibration_transform
        if ct.mu_calib is not None and ct.sigma_calib is not None:
            self._log(f"\nZ-score normalization fitted:")
            self._log(f"  mu_calib magnitude:    {np.linalg.norm(ct.mu_calib):.4f}")
            self._log(f"  sigma_calib magnitude: {np.linalg.norm(ct.sigma_calib):.4f}")
            if ct.rest_energy_threshold is not None:
                self._log(f"  rest energy gate:      {ct.rest_energy_threshold:.4f}")
            # Per-class residual in normalized space (lower = better alignment)
            common = set(ct.class_means_calib) & set(ct.class_means_train)
            if common:
                self._log("Per-class alignment (normalized residual — lower is better):")
                for cls in sorted(common):
                    norm_calib = (ct.class_means_calib[cls] - ct.mu_calib) / ct.sigma_calib
                    residual = np.linalg.norm(ct.class_means_train[cls] - norm_calib)
                    self._log(f"  {cls}: {residual:.3f}")

        self._log("\nDone! Click 'Apply Calibration to Prediction' to use it.")

    def _apply_calibration(self):
        if self.classifier is None or not self.classifier.calibration_transform.is_fitted:
            messagebox.showerror("Not Ready", "Run calibration first.")
            return

        app = self.winfo_toplevel()
        if isinstance(app, EMGApp):
            app.calibrated_classifier = self.classifier
            self.calib_applied_label.configure(
                text="Calibration applied! Disconnect, then go to Live Prediction.",
                text_color="green",
            )
            self._log("Calibration applied to Prediction page.")
            messagebox.showinfo(
                "Calibration Applied",
                "Session calibration is now active.\n\n"
                "Next steps:\n"
                "1. Click 'Disconnect' on this page\n"
                "2. Go to '5. Live Prediction'\n"
                "3. Connect to ESP32 there\n"
                "4. Choose Laptop inference mode\n"
                "5. Start Prediction — the calibrated model will be used automatically.",
            )


# =============================================================================
# LIVE PREDICTION PAGE
# =============================================================================

class PredictionPage(BasePage):
    """Page for live prediction demo."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Live Prediction",
            "Real-time gesture classification"
        )

        # State (MUST be initialized BEFORE creating UI elements)
        self.is_predicting = False
        self.is_connected = False
        self.classifier = None
        self.smoother = None
        self.stream = None
        self.data_queue = queue.Queue()
        self.inference_mode = "ESP32"  # "ESP32" or "Laptop"

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        # Model status
        self.status_frame = ctk.CTkFrame(self.content)
        self.status_frame.pack(fill="x", padx=20, pady=20)

        self.model_label = ctk.CTkLabel(
            self.status_frame,
            text="Checking for saved model...",
            font=ctk.CTkFont(size=14)
        )
        self.model_label.pack(pady=10)

        # Model file picker (for Laptop mode)
        self.model_picker_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.model_picker_frame.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(self.model_picker_frame, text="Model:", font=ctk.CTkFont(size=14)).pack(side="left")

        self.model_file_var = ctk.StringVar(value="No models found")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_picker_frame, variable=self.model_file_var,
            values=["No models found"], width=280,
        )
        self.model_dropdown.pack(side="left", padx=(10, 5))

        self.refresh_models_btn = ctk.CTkButton(
            self.model_picker_frame, text="⟳", width=30,
            command=self._refresh_model_list
        )
        self.refresh_models_btn.pack(side="left")

        # Initially hidden (only shown in Laptop mode)
        self.model_picker_frame.pack_forget()

        # Inference mode selector
        mode_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        mode_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(mode_frame, text="Inference:", font=ctk.CTkFont(size=14)).pack(side="left")

        self.mode_var = ctk.StringVar(value="ESP32")
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_frame, values=["ESP32", "Laptop"],
            variable=self.mode_var,
            command=self._on_mode_changed
        )
        self.mode_selector.pack(side="left", padx=(10, 0))

        self.mode_desc_label = ctk.CTkLabel(
            mode_frame,
            text="On-device inference (model baked into firmware)",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.mode_desc_label.pack(side="left", padx=(10, 0))

        # ESP32 Connection (hardware required)
        source_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        source_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(source_frame, text="ESP32 Connection:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        # Port selection
        port_select_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        port_select_frame.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(port_select_frame, text="Port:").pack(side="left")

        self.port_var = ctk.StringVar(value="Auto-detect")
        self.port_dropdown = ctk.CTkOptionMenu(
            port_select_frame, variable=self.port_var,
            values=["Auto-detect"], width=150
        )
        self.port_dropdown.pack(side="left", padx=(10, 5))

        self.refresh_ports_btn = ctk.CTkButton(
            port_select_frame, text="⟳", width=30,
            command=self._refresh_ports
        )
        self.refresh_ports_btn.pack(side="left")

        # Connection status and button
        connect_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        connect_frame.pack(fill="x", pady=(5, 0))

        self.connect_button = ctk.CTkButton(
            connect_frame, text="Connect",
            width=100, height=28,
            command=self._toggle_connection
        )
        self.connect_button.pack(side="left", padx=(0, 10))

        self.connection_status = ctk.CTkLabel(
            connect_frame, text="● Disconnected",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.connection_status.pack(side="left")

        # Start button
        self.start_button = ctk.CTkButton(
            self.content,
            text="Start Prediction",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            command=self.toggle_prediction
        )
        self.start_button.pack(pady=20)

        # Prediction display
        self.prediction_frame = ctk.CTkFrame(self.content)
        self.prediction_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.prediction_label = ctk.CTkLabel(
            self.prediction_frame,
            text="---",
            font=ctk.CTkFont(size=72, weight="bold")
        )
        self.prediction_label.pack(pady=30)

        self.confidence_bar = ctk.CTkProgressBar(self.prediction_frame, width=400, height=30)
        self.confidence_bar.pack(pady=10)
        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Confidence: ---%",
            font=ctk.CTkFont(size=18)
        )
        self.confidence_label.pack()

        # Simulated gesture indicator
        self.sim_label = ctk.CTkLabel(
            self.prediction_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.sim_label.pack(pady=10)

        # Smoothing info display
        self.smoothing_info_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Smoothing: EMA(0.7) + Majority(5) + Debounce(3)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.smoothing_info_label.pack()

        self.raw_label = ctk.CTkLabel(
            self.prediction_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.raw_label.pack(pady=5)

    def on_show(self):
        """Check model status when shown."""
        self._refresh_model_list()
        # If a calibrated classifier is available, surface it prominently
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp) and app.calibrated_classifier is not None:
            clf = app.calibrated_classifier
            self.model_label.configure(
                text=(
                    f"Calibrated model ready  ({clf.model_type.upper()}, "
                    f"{len(clf.label_names)} classes) — will be used in Laptop mode"
                ),
                text_color="green",
            )
        else:
            self.check_model()

    def _refresh_model_list(self):
        """Scan for saved models and populate the dropdown."""
        models = EMGClassifier.list_saved_models()
        if models:
            names = [p.name for p in models]
            self.model_dropdown.configure(values=names)
            # Default to most recent if current selection is invalid
            current = self.model_file_var.get()
            if current not in names:
                latest = max(models, key=lambda p: p.stat().st_mtime)
                self.model_file_var.set(latest.name)
        else:
            self.model_dropdown.configure(values=["No models found"])
            self.model_file_var.set("No models found")

    def _get_selected_model_path(self) -> Path | None:
        """Get the full path of the user-selected model file."""
        name = self.model_file_var.get()
        if name == "No models found":
            return None
        path = MODEL_DIR / name
        return path if path.exists() else None

    def check_model(self):
        """Check if a saved model exists (needed for Laptop mode)."""
        if self.inference_mode == "Laptop":
            # Show model picker in Laptop mode
            self.model_picker_frame.pack(fill="x", pady=(5, 0), after=self.model_label)
            model_path = self._get_selected_model_path()
            if model_path:
                self.model_label.configure(
                    text=f"Selected model: {model_path.name}",
                    text_color="green"
                )
                self.start_button.configure(state="normal")
            else:
                self.model_label.configure(
                    text="No saved models. Train a model first!",
                    text_color="orange"
                )
                self.start_button.configure(state="disabled")
        else:
            # ESP32 mode — hide model picker
            self.model_picker_frame.pack_forget()
            self.model_label.configure(
                text="ESP32 mode: model is baked into firmware",
                text_color="green"
            )
            self.start_button.configure(state="normal")

    def _on_mode_changed(self, mode: str):
        """Handle inference mode toggle."""
        self.inference_mode = mode
        if mode == "ESP32":
            self.mode_desc_label.configure(
                text="On-device inference (model baked into firmware)"
            )
        else:
            self.mode_desc_label.configure(
                text="Laptop inference (streams raw EMG, runs Python model)"
            )
        self._refresh_model_list()
        self.check_model()

    def toggle_prediction(self):
        """Start or stop prediction."""
        # Prevent rapid double-clicks from interfering
        if hasattr(self, '_toggling') and self._toggling:
            return

        self._toggling = True
        try:
            if self.is_predicting:
                self.stop_prediction()
            else:
                self.start_prediction()
        finally:
            # Reset flag after brief delay to prevent immediate re-trigger
            self.after(100, lambda: setattr(self, '_toggling', False))

    def _refresh_ports(self):
        """Scan and populate available serial ports."""
        ports = serial.tools.list_ports.comports()
        port_names = ["Auto-detect"] + [p.device for p in ports]
        self.port_dropdown.configure(values=port_names)

        if ports:
            self._update_connection_status("orange", f"Found {len(ports)} port(s)")
        else:
            self._update_connection_status("red", "No ports found")

    def _get_serial_port(self):
        """Get selected port, or None for auto-detect."""
        port = self.port_var.get()
        return None if port == "Auto-detect" else port

    def _update_connection_status(self, color: str, text: str):
        """Update the connection status indicator."""
        self.connection_status.configure(text=f"● {text}", text_color=color)

    def _toggle_connection(self):
        """Connect or disconnect from ESP32."""
        if self.is_connected:
            self._disconnect_device()
        else:
            self._connect_device()

    def _connect_device(self):
        """Connect to ESP32 with handshake."""
        port = self._get_serial_port()

        try:
            # Update UI to show connecting
            self._update_connection_status("orange", "Connecting...")
            self.connect_button.configure(state="disabled")
            self.update()  # Force UI update

            # Create stream and connect
            self.stream = RealSerialStream(port=port)
            device_info = self.stream.connect(timeout=5.0)

            # Success!
            self.is_connected = True
            self._update_connection_status("green", f"Connected ({device_info.get('device', 'ESP32')})")
            self.connect_button.configure(text="Disconnect", state="normal")

        except TimeoutError as e:
            messagebox.showerror(
                "Connection Timeout",
                f"Device did not respond within 5 seconds.\n\n"
                f"Check that:\n"
                f"• ESP32 is powered on\n"
                f"• Correct firmware is flashed\n"
                f"• USB cable is properly connected"
            )
            self._update_connection_status("red", "Timeout")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

        except Exception as e:
            error_msg = f"Failed to connect:\n{e}"
            if "Permission denied" in str(e) or "Resource busy" in str(e):
                error_msg += "\n\nThe port may still be in use. Wait a few seconds and try again."
            elif "FileNotFoundError" in str(type(e).__name__):
                error_msg += f"\n\nPort not found. Try refreshing the port list."

            messagebox.showerror("Connection Error", error_msg)
            self._update_connection_status("red", "Failed")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

    def _disconnect_device(self):
        """Disconnect from ESP32."""
        try:
            if self.stream:
                self.stream.disconnect()
                # Give OS time to release the port
                time.sleep(0.5)

            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")

        except Exception as e:
            messagebox.showwarning("Disconnect Warning", f"Error during disconnect: {e}")
            # Still mark as disconnected even if there was an error
            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")

    def start_prediction(self):
        """Start live prediction (dispatches based on inference mode)."""
        # Must be connected to ESP32
        if not self.is_connected or not self.stream:
            messagebox.showerror("Not Connected", "Please connect to ESP32 first.")
            return

        if self.inference_mode == "ESP32":
            self._start_esp32_prediction()
        else:
            self._start_laptop_prediction()

    def _start_esp32_prediction(self):
        """Start on-device inference (ESP32 runs LDA internally)."""
        print("[DEBUG] Starting ESP32 Prediction (On-Device)...")
        try:
            self.stream.start_predict()
            self.stream.running = True
        except Exception as e:
            messagebox.showerror("Start Error", f"Failed to start ESP32 prediction: {e}")
            return

        self.is_predicting = True
        self.start_button.configure(text="Stop Prediction", fg_color="red")
        self.connect_button.configure(state="disabled")
        self.mode_selector.configure(state="disabled")
        self.smoothing_info_label.configure(
            text="Smoothing: ESP32 firmware (EMA + Majority + Debounce)"
        )
        self.sim_label.configure(text="[ESP32 On-Device Inference]")
        self.raw_label.configure(text="")

        self.prediction_thread = threading.Thread(
            target=self._esp32_prediction_loop, daemon=True
        )
        self.prediction_thread.start()
        self.update_prediction_ui()

    def _start_laptop_prediction(self):
        """Start laptop-side inference (raw EMG stream + Python multi-model voting)."""
        print("[DEBUG] Starting Laptop Prediction...")

        # Prefer calibrated classifier from CalibrationPage if available
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp) and app.calibrated_classifier is not None:
            self.classifier = app.calibrated_classifier
            print(
                f"[Prediction] Using calibrated {self.classifier.model_type.upper()} "
                f"classifier (session-aligned)"
            )
        else:
            # Fall back to loading the user-selected model from disk
            model_path = self._get_selected_model_path()
            if not model_path:
                messagebox.showerror(
                    "No Model",
                    "No saved model found. Train a model first!\n\n"
                    "Tip: run '4. Calibrate' before predicting for better cross-session accuracy.",
                )
                return
            print(f"[DEBUG] Loading model: {model_path.name}")
            try:
                self.classifier = EMGClassifier.load(model_path)
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load model: {e}")
                return

        # Load ensemble model if available
        self._ensemble = None
        ensemble_path = Path(__file__).parent / 'models' / 'emg_ensemble.joblib'
        if ensemble_path.exists():
            try:
                import joblib
                self._ensemble = joblib.load(ensemble_path)
                print(f"[Prediction] Loaded ensemble model (4 LDAs)")
            except Exception as e:
                print(f"[Prediction] Ensemble load failed: {e}")

        # Load MLP weights if available
        self._mlp = None
        mlp_path = Path(__file__).parent / 'models' / 'emg_mlp_weights.npz'
        if mlp_path.exists():
            try:
                self._mlp = dict(np.load(mlp_path, allow_pickle=True))
                print(f"[Prediction] Loaded MLP weights (numpy)")
            except Exception as e:
                print(f"[Prediction] MLP load failed: {e}")

        # Report active models
        model_names = [self.classifier.model_type.upper()]
        if self._ensemble:
            model_names.append("Ensemble")
        if self._mlp:
            model_names.append("MLP")
        print(f"[Prediction] Active models: {' + '.join(model_names)} ({len(model_names)} total)")

        # Create smoother
        self.smoother = PredictionSmoother(
            label_names=self.classifier.label_names,
            probability_smoothing=0.7,
            majority_vote_window=5,
            debounce_count=4,
        )

        # Start raw EMG streaming from ESP32
        try:
            self.stream.start()
            self.stream.running = True
        except Exception as e:
            messagebox.showerror("Start Error", f"Failed to start raw streaming: {e}")
            return

        self.is_predicting = True
        self.start_button.configure(text="Stop Prediction", fg_color="red")
        self.connect_button.configure(state="disabled")
        self.mode_selector.configure(state="disabled")
        self.smoothing_info_label.configure(
            text="Smoothing: Python (EMA 0.7 + Majority 5 + Debounce 3)"
        )
        calib_active = self.classifier.calibration_transform.is_fitted
        mode_str = (
            f"[Laptop — {' + '.join(model_names)}"
            f"{' + Calibration' if calib_active else ''}]"
        )
        self.sim_label.configure(text=mode_str)

        self.prediction_thread = threading.Thread(
            target=self._laptop_prediction_loop, daemon=True
        )
        self.prediction_thread.start()
        self.update_prediction_ui()

    def stop_prediction(self):
        """Stop prediction (either mode)."""
        self.is_predicting = False
        if self.stream:
            self.stream.stop()

        self.start_button.configure(text="Start Prediction", fg_color=["#3B8ED0", "#1F6AA5"])
        self.prediction_label.configure(text="---", text_color="gray")
        self.confidence_label.configure(text="Confidence: ---%")
        self.confidence_bar.set(0)
        self.connect_button.configure(state="normal")
        self.mode_selector.configure(state="normal")
        self.sim_label.configure(text="")
        self.raw_label.configure(text="")

    def _esp32_prediction_loop(self):
        """Read JSON predictions from ESP32 on-device inference."""
        import json

        while self.is_predicting:
            try:
                line = self.stream.readline()
                if not line:
                    continue

                try:
                    line = line.strip()
                    if line.startswith('{'):
                        data = json.loads(line)

                        if "gesture" in data:
                            gesture = data["gesture"]
                            conf = float(data.get("conf", 0.0))
                            self.data_queue.put(('prediction', (gesture, conf)))

                        elif "status" in data:
                            print(f"[ESP32] {data}")

                except json.JSONDecodeError:
                    pass

            except Exception as e:
                if self.is_predicting:
                    print(f"ESP32 prediction loop error: {e}")
                    self.data_queue.put(('error', f"ESP32 error: {e}"))
                break

    def _run_ensemble(self, features: np.ndarray) -> np.ndarray:
        """Run ensemble prediction: 3 specialist LDAs → meta-LDA → probabilities."""
        ens = self._ensemble
        x_td = features[ens['td_idx']]
        x_fd = features[ens['fd_idx']]
        x_cc = features[ens['cc_idx']]
        p_td = ens['lda_td'].predict_proba([x_td])[0]
        p_fd = ens['lda_fd'].predict_proba([x_fd])[0]
        p_cc = ens['lda_cc'].predict_proba([x_cc])[0]
        x_meta = np.concatenate([p_td, p_fd, p_cc])
        return ens['meta_lda'].predict_proba([x_meta])[0]

    def _run_mlp(self, features: np.ndarray) -> np.ndarray:
        """Run MLP forward pass: Dense(32,relu) → Dense(16,relu) → Dense(5,softmax)."""
        m = self._mlp
        x = features.astype(np.float32)
        x = np.maximum(0, x @ m['w0'] + m['b0'])       # relu
        x = np.maximum(0, x @ m['w1'] + m['b1'])       # relu
        logits = x @ m['w2'] + m['b2']                  # softmax
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def _laptop_prediction_loop(self):
        """Parse raw EMG stream, window, extract features, multi-model vote."""
        parser = EMGParser(num_channels=NUM_CHANNELS)
        windower = Windower(
            window_size_ms=WINDOW_SIZE_MS,
            sample_rate=SAMPLING_RATE_HZ,
            hop_size_ms=HOP_SIZE_MS,
        )

        while self.is_predicting:
            try:
                line = self.stream.readline()
                if not line:
                    continue

                sample = parser.parse_line(line)
                if sample is None:
                    continue

                window = windower.add_sample(sample)
                if window is None:
                    continue

                window_data = window.to_numpy()

                # --- Base LDA prediction (includes energy gate + calibration) ---
                raw_label, proba_lda = self.classifier.predict(window_data)

                # If energy gate triggered rest, skip other models
                rest_gated = (raw_label == "rest" and proba_lda.max() == 1.0)

                if rest_gated:
                    avg_proba = proba_lda
                else:
                    # Extract calibrated features for ensemble/MLP
                    features_raw = self.classifier.feature_extractor.extract_features_window(window_data)
                    features = self.classifier.calibration_transform.apply(features_raw)

                    probas = [proba_lda]

                    # --- Ensemble ---
                    if self._ensemble:
                        try:
                            probas.append(self._run_ensemble(features))
                        except Exception:
                            pass

                    # --- MLP ---
                    if self._mlp:
                        try:
                            probas.append(self._run_mlp(features))
                        except Exception:
                            pass

                    avg_proba = np.mean(probas, axis=0)

                raw_label = self.classifier.label_names[int(np.argmax(avg_proba))]

                # Apply smoothing
                smoothed_label, smoothed_conf, _debug = self.smoother.update(raw_label, avg_proba)

                self.data_queue.put(('prediction', (smoothed_label, smoothed_conf)))

                # Show raw vs smoothed mismatch
                if raw_label != smoothed_label:
                    self.data_queue.put(('raw_info', f"raw: {raw_label}"))
                else:
                    self.data_queue.put(('raw_info', ""))

            except Exception as e:
                if self.is_predicting:
                    import traceback
                    traceback.print_exc()
                    self.data_queue.put(('error', f"Prediction error: {e}"))
                break

    def update_prediction_ui(self):
        """Update UI from queue."""
        try:
            while True:
                msg_type, data = self.data_queue.get_nowait()
                
                if msg_type == 'prediction':
                    label, conf = data

                    # Update label
                    self.prediction_label.configure(
                        text=label.upper(),
                        text_color=get_gesture_color(label)
                    )

                    # Update confidence
                    self.confidence_label.configure(text=f"Confidence: {conf*100:.1f}%")
                    self.confidence_bar.set(conf)

                elif msg_type == 'raw_info':
                    # Show raw vs smoothed mismatch (laptop mode only)
                    self.raw_label.configure(text=data, text_color="orange" if data else "gray")

                elif msg_type == 'sim_gesture':
                    self.sim_label.configure(text=f"[Simulating: {data}]")

                elif msg_type == 'error':
                    # Show error and stop prediction
                    self._update_connection_status("red", "Disconnected")
                    messagebox.showerror("Prediction Error", data)
                    self.stop_prediction()
                    return

                elif msg_type == 'connection_status':
                    # Update connection indicator
                    color, text = data
                    self._update_connection_status(color, text)
                    # Also update sim_label to indicate real hardware
                    if text == "Connected":
                        self.sim_label.configure(text="[Real ESP32 Hardware]")

        except queue.Empty:
            pass

        if self.is_predicting:
            self.after(50, self.update_prediction_ui)

    def on_hide(self):
        """Stop when leaving page."""
        if self.is_predicting:
            self.stop_prediction()

    def stop(self):
        """Stop everything."""
        self.is_predicting = False
        if self.stream:
            self.stream.stop()


# =============================================================================
# VISUALIZATION PAGE
# =============================================================================

class VisualizationPage(BasePage):
    """Page for LDA visualization."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "LDA Visualization",
            "Visualize decision boundaries and feature space"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        # Controls
        self.controls = ctk.CTkFrame(self.content)
        self.controls.pack(fill="x", padx=20, pady=20)

        self.generate_button = ctk.CTkButton(
            self.controls,
            text="Generate Visualizations",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.generate_plots
        )
        self.generate_button.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(self.controls, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=20)

        # Plot area
        self.plot_frame = ctk.CTkFrame(self.content)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.canvas = None

    def generate_plots(self):
        """Generate LDA visualization plots."""
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            messagebox.showwarning("No Data", "No sessions found. Collect data first!")
            return

        self.status_label.configure(text="Loading data...")
        self.generate_button.configure(state="disabled")

        # Run in thread
        thread = threading.Thread(target=self._generate_thread, daemon=True)
        thread.start()

    def _generate_thread(self):
        """Generate plots in background."""
        try:
            storage = SessionStorage()
            X, y, _trial_ids, session_indices, label_names, _ = storage.load_all_for_training()

            self.after(0, lambda: self.status_label.configure(text="Extracting features..."))

            # Extract features matching the training pipeline
            extractor = EMGFeatureExtractor(
                channels=HAND_CHANNELS, expanded=True,
                cross_channel=True, bandpass=True,
            )
            X_features = extractor.extract_features_batch(X)

            # Apply per-session z-score normalization (matches training pipeline)
            for sid in np.unique(session_indices):
                mask = session_indices == sid
                X_sess = X_features[mask]
                y_sess = y[mask]
                class_means = [X_sess[y_sess == c].mean(axis=0)
                               for c in np.unique(y_sess)]
                balanced_mean = np.mean(class_means, axis=0)
                std = X_sess.std(axis=0)
                std[std < 1e-12] = 1.0
                X_features[mask] = (X_sess - balanced_mean) / std

            lda = LinearDiscriminantAnalysis()
            lda.fit(X_features, y)
            X_lda = lda.transform(X_features)

            self.after(0, lambda: self.status_label.configure(text="Creating plots..."))

            n_classes = len(label_names)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_classes))

            # Create figure
            fig = Figure(figsize=(12, 5), dpi=100, facecolor='#2b2b2b')

            # Plot 1: LDA Feature Space with Decision Boundaries
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_facecolor('#2b2b2b')
            ax1.tick_params(colors='white')

            # Create mesh grid for decision boundaries
            if X_lda.shape[1] >= 2:
                x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
                y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1

                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 200),
                    np.linspace(y_min, y_max, 200)
                )

                # Train a classifier on the 2D LDA space for visualization
                lda_2d = LinearDiscriminantAnalysis()
                lda_2d.fit(X_lda[:, :2], y)
                Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot decision regions (filled contours)
                ax1.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, n_classes, 1),
                            colors=[colors[i] for i in range(n_classes)])

                # Plot decision boundaries (lines)
                ax1.contour(xx, yy, Z, colors='white', linewidths=1.5, alpha=0.8)

            # Plot data points
            for i, label in enumerate(label_names):
                mask = y == i
                ax1.scatter(
                    X_lda[mask, 0],
                    X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                    c=[colors[i]], label=label, s=50, alpha=0.9,
                    edgecolors='white', linewidth=0.5
                )

            ax1.set_xlabel('LDA Component 1', color='white')
            ax1.set_ylabel('LDA Component 2', color='white')
            ax1.set_title('LDA Decision Boundaries', color='white', fontsize=14)
            ax1.legend(facecolor='#2b2b2b', labelcolor='white', loc='upper right')
            for spine in ax1.spines.values():
                spine.set_color('white')

            # Plot 2: Class distributions
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_facecolor('#2b2b2b')
            ax2.tick_params(colors='white')

            for i, label in enumerate(label_names):
                mask = y == i
                ax2.hist(X_lda[mask, 0], bins=20, alpha=0.6, label=label, color=colors[i])

            ax2.set_xlabel('LDA Component 1', color='white')
            ax2.set_ylabel('Count', color='white')
            ax2.set_title('Class Distributions', color='white', fontsize=14)
            ax2.legend(facecolor='#2b2b2b', labelcolor='white')
            for spine in ax2.spines.values():
                spine.set_color('white')

            fig.tight_layout()

            # Display in GUI
            self.after(0, lambda: self._show_plot(fig))
            self.after(0, lambda: self.status_label.configure(text="Done!"))

        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))

        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def _show_plot(self, fig):
        """Show the plot in the GUI."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = EMGApp()
    app.mainloop()
