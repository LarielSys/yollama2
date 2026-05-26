"""
YOLLama LiveCam — Road Damage Detection
Real-time YOLO camera view that processes a folder of road images
as a simulated dashcam feed, with live annotated detection overlay.

Architecture mirrors LarielTraderPlatform V2:
  Left panel  — Image queue, playback controls, YOLO settings
  Center panel — Live YOLO camera feed (annotated frames)
  Right panel  — Detection log, class breakdown, Ollama analysis, stats
"""

import io
import os
import sys
import json
import time
import base64
import threading
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Optional: hit the Docker Ollama service for AI interpretation
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colour constants matching Lariel's dark theme
# ---------------------------------------------------------------------------
BG = "#1e1e1e"
SURFACE = "#2d2d2d"
DARK = "#1a1a1a"
DARKER = "#0d0d0d"
GREEN = "#00ff00"
CYAN = "#00ffff"
ORANGE = "#ffaa00"
RED = "#ff4444"
WHITE = "#ffffff"
GREY = "#888888"
ACCENT = "#6366f1"

# Damage class colours for the detection log
CLASS_COLOURS = {
    "D00": "#00ccff",  # cyan  — longitudinal/transverse cracks
    "D10": "#ff9900",  # orange — reflection cracks
    "D20": "#ff3333",  # red    — fatigue/alligator cracking
    "D40": "#cc33ff",  # purple — potholes and patching
    "baches_r": "#cc33ff",  # pothole-specific single-class Rocher model
    "roderas": "#ffaa00",   # rutting-specific single-class Rocher model
}

CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal/transverse cracks",
    "D10": "Reflection cracks (concrete joints)",
    "D20": "Fatigue/alligator cracking",
    "D40": "Potholes and patching",
    "baches_r": "Baches (Rocher pothole class)",
    "roderas": "Roderas / Rutting",
}

CANONICAL_CONNECTORS = {
    "baches_r": "D40",
}

SEVERITY_THRESHOLDS = {"low": 0.40, "medium": 0.60, "high": 0.80}


def severity_label(conf: float) -> str:
    if conf >= SEVERITY_THRESHOLDS["high"]:
        return "CRITICAL"
    elif conf >= SEVERITY_THRESHOLDS["medium"]:
        return "HIGH"
    elif conf >= SEVERITY_THRESHOLDS["low"]:
        return "MEDIUM"
    return "LOW"


def severity_colour(conf: float) -> str:
    if conf >= SEVERITY_THRESHOLDS["high"]:
        return RED
    elif conf >= SEVERITY_THRESHOLDS["medium"]:
        return ORANGE
    elif conf >= SEVERITY_THRESHOLDS["low"]:
        return "#eab308"
    return GREEN


# ── RDD2022 reference similarity (US / Czech) ───────────────────────────────

_D_CANONICAL = {"D00", "D10", "D20", "D40"}
_ROADREADER_TO_D = {
    "asf_agrietamiento_fatiga": "D20",
    "asf_agrietamiento_bloque": "D20",
    "asf_grietas_longitudinales_transversales": "D00",
    "asf_grieta_borde": "D00",
    "asf_grieta_reflexion_junta_losa_concreto": "D10",
    "asf_grietas_parabolicas_deslizamiento": "D20",
    "asf_hundimientos_asentamientos": "D40",
    "asf_bacheo_superficial": "D40",
    "asf_baches": "D40",
}
_REF_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _img_to_vec(pil_img: Image.Image) -> "np.ndarray | None":
    try:
        arr = np.asarray(
            pil_img.convert("L").resize((64, 64), Image.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
        arr -= float(arr.mean())
        norm = float(np.linalg.norm(arr))
        return (arr / norm).astype(np.float32) if norm > 1e-6 else None
    except Exception:
        return None


def _read_json_labels(json_path: Path) -> set:
    if not json_path.is_file():
        return set()
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    mapped: set = set()
    for shape in data.get("shapes", []):
        lbl = str(shape.get("label", "")).strip()
        d = _ROADREADER_TO_D.get(lbl, lbl if lbl in _D_CANONICAL else "")
        if d in _D_CANONICAL:
            mapped.add(d)
    return mapped


def build_ref_index(root: Path, max_images: int) -> dict:
    if not root.is_dir():
        return {"available": False, "count": 0, "paths": [], "labels": [], "matrix": None}
    paths: list = []
    labels: list = []
    vecs: list = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _REF_EXTS:
            continue
        try:
            with Image.open(p) as img:
                vec = _img_to_vec(img.convert("RGB"))
        except Exception:
            continue
        if vec is None:
            continue
        paths.append(str(p))
        labels.append(_read_json_labels(p.with_suffix(".json")))
        vecs.append(vec)
        if max_images > 0 and len(paths) >= max_images:
            break
    if not vecs:
        return {"available": False, "count": 0, "paths": [], "labels": [], "matrix": None}
    return {
        "available": True,
        "count": len(vecs),
        "paths": paths,
        "labels": labels,
        "matrix": np.vstack(vecs).astype(np.float32),
    }


def ref_best_match(pil_img: Image.Image, hint: str, indices: list) -> tuple:
    vec = _img_to_vec(pil_img)
    if vec is None:
        return hint, 0.0
    best_cls, best_score = hint, -1.0
    for idx in indices:
        if not idx.get("available") or idx["matrix"] is None:
            continue
        scores = (idx["matrix"] @ vec).tolist()
        candidates = [
            (s, i) for i, (s, lbls) in enumerate(zip(scores, idx["labels"]))
            if hint in lbls
        ]
        if not candidates:
            candidates = [(s, i) for i, s in enumerate(scores)]
        if not candidates:
            continue
        top_score, top_i = max(candidates, key=lambda x: x[0])
        top_labels = idx["labels"][top_i]
        if top_score > best_score:
            best_score = top_score
            best_cls = sorted(top_labels)[0] if top_labels else hint
    return best_cls, best_score


# ───────────────────────────────────────────────────────────────────────────
class LiveCamApp:
    """Main application — mirrors LarielPlatform layout."""

    # ── init ──────────────────────────────────────────────────────────────
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLLama LiveCam — Road Damage Detection")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG)

        # YOLO model (loaded in-process for real-time speed)
        self.model: YOLO | None = None
        self.models_dir = Path(__file__).resolve().parent.parent / "yolo-service" / "models"
        self.available_model_files = self._available_models()
        self.selected_model_name = self._default_model_name()
        self.model_path = str(self.models_dir / self.selected_model_name)

        # Image feed state
        self.source_mode: str = "folder"
        self.image_folder: str | None = None
        self.image_files: list[Path] = []
        self.current_index: int = 0
        self.playing: bool = False
        self.frame_delay_ms: int = 1500  # ms between frames
        self.confidence: float = 0.25
        self.camera_index: int = 0
        self.cap: cv2.VideoCapture | None = None

        # Stats
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.class_counts: dict[str, int] = {}
        self.start_time: float | None = None

        # Ollama
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Use a tagged default model name because many Ollama installs require exact tags.
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

        # OpenAI
        self.openai_api_key: str = ""

        # Reference DB (US / Czech RDD2022)
        self.ref_indices: list = []
        self.ref_enabled: bool = False
        self.ref_fallback_conf: float = 0.30
        self._ref_building: bool = False

        # Processing state (sequential frame handling — no overlapping threads)
        self._processing = False
        self._preload_buffer: dict[int, Image.Image] = {}
        self._preload_ahead = 10

        # Current annotated frame (kept for resize)
        self._current_photo: ImageTk.PhotoImage | None = None

        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_model()

    # ── GUI skeleton ──────────────────────────────────────────────────────
    def _build_gui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        main.grid_columnconfigure(0, weight=0, minsize=300)
        main.grid_columnconfigure(1, weight=1, minsize=700)
        main.grid_columnconfigure(2, weight=0, minsize=340)
        main.grid_rowconfigure(0, weight=1)

        self._build_left(main)
        self._build_center(main)
        self._build_right(main)

    # ── LEFT PANEL ────────────────────────────────────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=SURFACE, relief=tk.RAISED, borderwidth=2)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Title
        tk.Label(left, text="IMAGE FEED", font=("Arial", 14, "bold"),
                 bg=SURFACE, fg=GREEN).pack(pady=10)

        ctrl = tk.Frame(left, bg=SURFACE)
        ctrl.pack(fill=tk.X, padx=10)

        # Folder picker
        tk.Button(ctrl, text="Load Image Folder…", command=self._pick_folder,
                  bg="#0055aa", fg=WHITE, font=("Arial", 10, "bold"),
                  relief=tk.RAISED, borderwidth=2).pack(fill=tk.X, pady=5)

        webcam_row = tk.Frame(ctrl, bg=SURFACE)
        webcam_row.pack(fill=tk.X, pady=5)

        tk.Label(webcam_row, text="Camera:", bg=SURFACE, fg=WHITE,
             font=("Arial", 9)).pack(side=tk.LEFT)
        self.camera_index_var = tk.StringVar(value="0")
        tk.Entry(webcam_row, textvariable=self.camera_index_var,
             width=5, bg=DARKER, fg=GREEN, insertbackground=GREEN,
             font=("Courier", 9), relief=tk.SUNKEN, borderwidth=2).pack(side=tk.LEFT, padx=5)
        tk.Button(webcam_row, text="Use Webcam", command=self._enable_webcam,
              bg="#007744", fg=WHITE, font=("Arial", 9, "bold"),
              relief=tk.RAISED, borderwidth=2).pack(side=tk.LEFT, padx=3)

        # Model picker
        tk.Label(ctrl, text="YOLO Model:", bg=SURFACE, fg=WHITE,
                 font=("Arial", 9)).pack(pady=(8, 2), anchor=tk.W)
        self.model_var = tk.StringVar(value=self.selected_model_name)
        self.model_combo = ttk.Combobox(
            ctrl,
            textvariable=self.model_var,
            values=self.available_model_files,
            state="readonly",
            font=("Arial", 9),
        )
        self.model_combo.pack(fill=tk.X, pady=2)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        tk.Button(ctrl, text="Load Selected Model", command=self._load_selected_model,
                  bg="#7a4f00", fg=WHITE, font=("Arial", 9, "bold"),
                  relief=tk.RAISED, borderwidth=2).pack(fill=tk.X, pady=4)

        self.folder_label = tk.Label(ctrl, text="No folder loaded", bg=SURFACE,
                                     fg=GREY, font=("Arial", 8), wraplength=270)
        self.folder_label.pack(pady=2)

        self.count_label = tk.Label(ctrl, text="", bg=SURFACE, fg=WHITE,
                                    font=("Arial", 9))
        self.count_label.pack(pady=2)

        # Playback controls
        sep = tk.Frame(ctrl, bg=ACCENT, height=2)
        sep.pack(fill=tk.X, pady=10)

        tk.Label(ctrl, text="PLAYBACK", font=("Arial", 11, "bold"),
                 bg=SURFACE, fg=CYAN).pack(pady=5)

        btn_row = tk.Frame(ctrl, bg=SURFACE)
        btn_row.pack(fill=tk.X, pady=5)

        self.play_btn = tk.Button(btn_row, text="▶  Play", command=self._toggle_play,
                                  bg="#00aa00", fg=WHITE, font=("Arial", 11, "bold"),
                                  relief=tk.RAISED, borderwidth=3, width=10)
        self.play_btn.pack(side=tk.LEFT, padx=3)

        tk.Button(btn_row, text="⏭", command=self._next_frame,
                  bg="#555555", fg=WHITE, font=("Arial", 11), width=4).pack(side=tk.LEFT, padx=3)

        tk.Button(btn_row, text="⏮", command=self._prev_frame,
                  bg="#555555", fg=WHITE, font=("Arial", 11), width=4).pack(side=tk.LEFT, padx=3)

        # Speed slider
        tk.Label(ctrl, text="Frame Delay (ms):", bg=SURFACE, fg=WHITE,
                 font=("Arial", 9)).pack(pady=(10, 2), anchor=tk.W)

        self.speed_var = tk.IntVar(value=self.frame_delay_ms)
        speed_slider = tk.Scale(ctrl, from_=200, to=5000, orient=tk.HORIZONTAL,
                                variable=self.speed_var, bg=SURFACE, fg=WHITE,
                                troughcolor=DARKER, highlightthickness=0,
                                font=("Arial", 8), command=self._on_speed_change)
        speed_slider.pack(fill=tk.X, pady=2)

        # Confidence slider
        tk.Label(ctrl, text="Confidence Threshold:", bg=SURFACE, fg=WHITE,
                 font=("Arial", 9)).pack(pady=(10, 2), anchor=tk.W)

        self.conf_var = tk.DoubleVar(value=self.confidence)
        self.conf_label = tk.Label(ctrl, text=f"{self.confidence:.2f}", bg=SURFACE,
                                   fg=GREEN, font=("Courier", 10, "bold"))
        self.conf_label.pack()

        conf_slider = tk.Scale(ctrl, from_=0.05, to=0.95, resolution=0.05,
                               orient=tk.HORIZONTAL, variable=self.conf_var,
                               bg=SURFACE, fg=WHITE, troughcolor=DARKER,
                               highlightthickness=0, font=("Arial", 8),
                               command=self._on_conf_change)
        conf_slider.pack(fill=tk.X, pady=2)

        # OpenAI API key
        sep_ai = tk.Frame(ctrl, bg=ACCENT, height=2)
        sep_ai.pack(fill=tk.X, pady=10)

        tk.Label(ctrl, text="OPENAI API KEY", font=("Arial", 11, "bold"),
                 bg=SURFACE, fg="#74aa9c").pack(pady=5)

        self.api_key_var = tk.StringVar()
        self.api_key_entry = tk.Entry(ctrl, textvariable=self.api_key_var,
                                      bg=DARKER, fg=GREEN, insertbackground=GREEN,
                                      font=("Courier", 8), show="*",
                                      relief=tk.SUNKEN, borderwidth=2)
        self.api_key_entry.pack(fill=tk.X, pady=2)
        self.api_key_entry.bind("<FocusOut>", self._on_api_key_change)
        self.api_key_entry.bind("<Return>", self._on_api_key_change)

        self.api_key_status = tk.Label(ctrl, text="No key set — using Ollama",
                                       bg=SURFACE, fg=GREY, font=("Arial", 7))
        self.api_key_status.pack(pady=2)

        # Image list (scrollable)
        sep2 = tk.Frame(ctrl, bg=ACCENT, height=2)
        sep2.pack(fill=tk.X, pady=10)

        tk.Label(ctrl, text="IMAGE QUEUE", font=("Arial", 11, "bold"),
                 bg=SURFACE, fg=ORANGE).pack(pady=5)

        # ── Reference DB toggle ────────────────────────────────────────
        sep_ref = tk.Frame(ctrl, bg="#6633aa", height=2)
        sep_ref.pack(fill=tk.X, pady=8)

        tk.Label(ctrl, text="US/CZECH REFERENCE DB", font=("Arial", 10, "bold"),
                 bg=SURFACE, fg="#bb88ff").pack(pady=3)

        ref_ctrl = tk.Frame(ctrl, bg=SURFACE)
        ref_ctrl.pack(fill=tk.X)

        self.ref_enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ref_ctrl,
            text="Enable cross-check (low-conf frames)",
            variable=self.ref_enabled_var,
            command=self._on_ref_toggle,
            bg=SURFACE, fg=WHITE, selectcolor=DARK,
            activebackground=SURFACE,
            font=("Arial", 8),
        ).pack(anchor=tk.W)

        self.ref_status_label = tk.Label(
            ref_ctrl, text="Not loaded",
            bg=SURFACE, fg=GREY, font=("Arial", 7), wraplength=270,
        )
        self.ref_status_label.pack(anchor=tk.W, pady=2)

        list_frame = tk.Frame(left, bg=SURFACE)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.image_listbox = tk.Listbox(list_frame, bg=DARKER, fg=GREEN,
                                        font=("Courier", 8), selectbackground=ACCENT,
                                        selectforeground=WHITE, activestyle="none")
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                 command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)

    # ── CENTER PANEL — THE YOLO CAMERA ────────────────────────────────────
    def _build_center(self, parent):
        center = tk.Frame(parent, bg="#000000", relief=tk.SUNKEN, borderwidth=3)
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Top bar — camera info
        self.cam_bar = tk.Frame(center, bg="#111111", height=30)
        self.cam_bar.pack(fill=tk.X)
        self.cam_bar.pack_propagate(False)

        self.cam_status = tk.Label(self.cam_bar, text="●  YOLO CAMERA — IDLE",
                                   font=("Courier", 10, "bold"), bg="#111111", fg=GREY)
        self.cam_status.pack(side=tk.LEFT, padx=10)

        self.cam_fps = tk.Label(self.cam_bar, text="",
                                font=("Courier", 9), bg="#111111", fg=CYAN)
        self.cam_fps.pack(side=tk.RIGHT, padx=10)

        self.cam_file = tk.Label(self.cam_bar, text="",
                                 font=("Courier", 9), bg="#111111", fg=ORANGE)
        self.cam_file.pack(side=tk.RIGHT, padx=10)

        # Canvas for the annotated frames
        self.canvas = tk.Canvas(center, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder
        self.canvas.create_text(
            400, 300, text="YOLO CAMERA\n\nLoad images or connect a webcam, then press Play",
            fill="#555555", font=("Arial", 24, "bold"), justify=tk.CENTER,
            tags="placeholder",
        )

    # ── RIGHT PANEL — 4 sections ──────────────────────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=SURFACE)
        right.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        for i in range(4):
            right.grid_rowconfigure(i, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_detection_section(right, 0)
        self._build_class_section(right, 1)
        self._build_analysis_section(right, 2)
        self._build_stats_section(right, 3)

    def _build_detection_section(self, parent, row):
        """Section 1: Live Detection Log"""
        frame = tk.Frame(parent, bg=DARK, relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=0, sticky="nsew", padx=2, pady=2)

        tk.Label(frame, text="🔍 DETECTION LOG", font=("Arial", 11, "bold"),
                 bg=DARK, fg=CYAN).pack(pady=5)

        self.det_text = scrolledtext.ScrolledText(frame, height=6, width=38,
                                                  bg=DARKER, fg=CYAN,
                                                  font=("Courier", 8), wrap=tk.WORD)
        self.det_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.det_text.insert(tk.END, "Waiting for camera feed…\n")
        self.det_text.config(state=tk.DISABLED)

    def _build_class_section(self, parent, row):
        """Section 2: Class Breakdown (like Candle Translation)"""
        frame = tk.Frame(parent, bg=DARK, relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=0, sticky="nsew", padx=2, pady=2)

        tk.Label(frame, text="📊 CLASS BREAKDOWN", font=("Arial", 11, "bold"),
                 bg=DARK, fg=ORANGE).pack(pady=5)

        self.class_text = scrolledtext.ScrolledText(frame, height=6, width=38,
                                                    bg=DARKER, fg=ORANGE,
                                                    font=("Courier", 8), wrap=tk.WORD)
        self.class_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.class_text.insert(tk.END, "No detections yet.\n")
        self.class_text.config(state=tk.DISABLED)

    def _build_analysis_section(self, parent, row):
        """Section 3: AI Analysis (Ollama interpretation)"""
        frame = tk.Frame(parent, bg=DARK, relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=0, sticky="nsew", padx=2, pady=2)

        tk.Label(frame, text="💡 AI ANALYSIS", font=("Arial", 11, "bold"),
                 bg=DARK, fg=GREEN).pack(pady=5)

        self.analysis_text = scrolledtext.ScrolledText(frame, height=6, width=38,
                                                       bg=DARKER, fg=GREEN,
                                                       font=("Courier", 8),
                                                       wrap=tk.WORD)
        self.analysis_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.analysis_text.insert(tk.END, "AI analysis will appear here.\n")
        self.analysis_text.config(state=tk.DISABLED)

    def _build_stats_section(self, parent, row):
        """Section 4: Session Stats (like Traders panel)"""
        frame = tk.Frame(parent, bg=DARK, relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=0, sticky="nsew", padx=2, pady=2)

        tk.Label(frame, text="📈 SESSION STATS", font=("Arial", 11, "bold"),
                 bg=DARK, fg=WHITE).pack(pady=5)

        stats_container = tk.Frame(frame, bg=DARK)
        stats_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # YOLO model info
        model_frame = tk.LabelFrame(stats_container, text="YOLO Model",
                                    bg=DARKER, fg=GREEN, font=("Arial", 9, "bold"))
        model_frame.pack(fill=tk.X, pady=3)

        self.model_info_label = tk.Label(model_frame,
                                         text="Loading model…",
                                         bg=DARKER, fg=GREEN,
                                         font=("Courier", 7), justify=tk.LEFT,
                                         wraplength=290)
        self.model_info_label.pack(padx=3, pady=2)

        # Session stats
        session_frame = tk.LabelFrame(stats_container, text="Session",
                                      bg=DARKER, fg=ORANGE, font=("Arial", 9, "bold"))
        session_frame.pack(fill=tk.X, pady=3)

        self.stats_label = tk.Label(session_frame,
                                    text="Frames: 0\nDetections: 0\nElapsed: 0s",
                                    bg=DARKER, fg=ORANGE,
                                    font=("Courier", 7), justify=tk.LEFT,
                                    wraplength=290)
        self.stats_label.pack(padx=3, pady=2)

        # Class histogram canvas
        hist_frame = tk.LabelFrame(stats_container, text="Detections by Class",
                                   bg=DARKER, fg=CYAN, font=("Arial", 9, "bold"))
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=3)

        self.hist_canvas = tk.Canvas(hist_frame, bg="#000000", height=60,
                                     highlightthickness=1,
                                     highlightbackground="#333333")
        self.hist_canvas.pack(fill=tk.X, padx=3, pady=3)

    # ── Model loading ─────────────────────────────────────────────────────
    def _available_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted([p.name for p in self.models_dir.glob("*.pt")])

    def _default_model_name(self) -> str:
        preferred = [
            "daka_rocher_model_v3.0.pt",
            "road_damage.pt",
        ]
        for name in preferred:
            if (self.models_dir / name).exists():
                return name
        models = self._available_models()
        return models[0] if models else "road_damage.pt"

    def _on_model_selected(self, _event=None):
        self._load_selected_model()

    def _load_selected_model(self):
        selected = self.model_var.get().strip()
        if not selected:
            messagebox.showwarning("Model", "No model selected.")
            return
        if self.playing:
            self.playing = False
            self.play_btn.config(text="▶  Play", bg="#00aa00")
            self.cam_status.config(text="●  YOLO CAMERA — PAUSED", fg=ORANGE)
        self._load_model(selected)

    def _load_model(self, model_name: str | None = None):
        if model_name:
            self.selected_model_name = Path(model_name).name
        self.model_path = str(self.models_dir / self.selected_model_name)

        def _load():
            try:
                if not Path(self.model_path).exists():
                    raise FileNotFoundError(f"Model not found: {self.model_path}")
                self.model = YOLO(self.model_path)
                # Warm-up
                dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
                self.model.predict(source=dummy, conf=0.25, imgsz=640, verbose=False, half=True)
                device = str(next(self.model.model.parameters()).device)
                names = ", ".join(self.model.names.values())
                self.root.after(0, lambda: self.model_info_label.config(
                    text=f"Model: {Path(self.model_path).name}\nDevice: {device}\nClasses: {names}"))
                self.root.after(0, lambda: self._log_det(
                    f"✓ Model loaded: {Path(self.model_path).name} on {device}\n"))
            except Exception as e:
                self.root.after(0, lambda: self.model_info_label.config(
                    text=f"LOAD FAILED:\n{e}"))
                self.root.after(0, lambda: self._log_det(f"✗ Model error: {e}\n"))

        threading.Thread(target=_load, daemon=True).start()

    # ── Folder handling ───────────────────────────────────────────────────
    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select Road Image Folder")
        if not folder:
            return
        self._release_camera()
        self.source_mode = "folder"
        self.image_folder = folder
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.image_files = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
        )
        self.current_index = 0
        self.folder_label.config(text=folder, fg=WHITE)
        self.count_label.config(text=f"{len(self.image_files)} images found")

        # Populate listbox
        self.image_listbox.delete(0, tk.END)
        for p in self.image_files:
            self.image_listbox.insert(tk.END, p.name)
        if self.image_files:
            self.image_listbox.selection_set(0)
            self.image_listbox.see(0)

    def _enable_webcam(self):
        if self.model is None:
            messagebox.showwarning("No Model", "YOLO model is still loading.")
            return

        try:
            camera_index = int(self.camera_index_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Camera", "Camera index must be a number, like 0 or 1.")
            return

        self._release_camera()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open webcam {camera_index}.")
            return

        self.cap = cap
        self.camera_index = camera_index
        self.source_mode = "webcam"
        self.image_folder = None
        self.image_files = []
        self.current_index = 0
        self.folder_label.config(text=f"Webcam {camera_index} connected", fg=GREEN)
        self.count_label.config(text="Live camera feed")
        self.image_listbox.delete(0, tk.END)
        self.cam_status.config(text=f"●  YOLO CAMERA — CAM {camera_index} READY", fg=CYAN)

        if not self.playing and not self._processing:
            self._process_current_frame()

    def _on_list_select(self, _event):
        sel = self.image_listbox.curselection()
        if sel:
            self.current_index = sel[0]
            if not self.playing and not self._processing:
                self._process_current_frame()

    def _on_ref_toggle(self):
        self.ref_enabled = self.ref_enabled_var.get()
        if self.ref_enabled and not self.ref_indices and not self._ref_building:
            self._build_ref_indices()

    def _build_ref_indices(self):
        self._ref_building = True
        self.ref_status_label.config(text="Building index\u2026", fg=ORANGE)

        def _build():
            us_path = Path(r"C:\RDD2022_labelme\UnitedStates\train")
            cz_path = Path(r"C:\RDD2022_labelme\Czech\train")
            indices = []
            for name, path in [("US", us_path), ("Czech", cz_path)]:
                self.root.after(0, lambda n=name: self.ref_status_label.config(
                    text=f"Loading {n} index\u2026", fg=ORANGE))
                idx = build_ref_index(path, 2000)
                idx["name"] = name
                indices.append(idx)
            self.ref_indices = indices
            self._ref_building = False
            total = sum(i["count"] for i in indices)
            self.root.after(0, lambda: self.ref_status_label.config(
                text=f"Ready \u2014 {total} ref images loaded", fg=GREEN))

        threading.Thread(target=_build, daemon=True).start()

    # ── Playback ──────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.source_mode == "folder" and not self.image_files:
            messagebox.showwarning("No Images", "Load an image folder first.")
            return
        if self.source_mode == "webcam" and self.cap is None:
            messagebox.showwarning("No Camera", "Connect a webcam first.")
            return
        if self.model is None:
            messagebox.showwarning("No Model", "YOLO model is still loading.")
            return

        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸  Pause", bg="#aa0000")
            self.cam_status.config(text="●  YOLO CAMERA — LIVE", fg=GREEN)
            self.start_time = self.start_time or time.time()
            self._play_loop()
        else:
            self.play_btn.config(text="▶  Play", bg="#00aa00")
            self.cam_status.config(text="●  YOLO CAMERA — PAUSED", fg=ORANGE)

    def _play_loop(self):
        if not self.playing:
            return
        if self._processing:
            self.root.after(50, self._play_loop)
            return
        self._process_current_frame(auto_advance=True)

    def _next_frame(self):
        if self.source_mode != "folder" or not self.image_files or self._processing:
            return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self._process_current_frame()
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)

    def _prev_frame(self):
        if self.source_mode != "folder" or not self.image_files or self._processing:
            return
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self._process_current_frame()
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)

    def _on_speed_change(self, val):
        self.frame_delay_ms = int(float(val))

    def _on_conf_change(self, val):
        self.confidence = float(val)
        self.conf_label.config(text=f"{self.confidence:.2f}")

    def _on_api_key_change(self, _event=None):
        key = self.api_key_var.get().strip()
        self.openai_api_key = key
        if key:
            masked = key[:3] + "…" + key[-4:]
            self.api_key_status.config(text=f"GPT-4o-mini active ({masked})", fg=GREEN)
        else:
            self.api_key_status.config(text="No key set — using Ollama", fg=GREY)

    # ── Core: process a single frame ──────────────────────────────────────
    def _process_current_frame(self, auto_advance=False):
        if self.model is None or self._processing:
            return
        if self.source_mode == "folder" and not self.image_files:
            return
        if self.source_mode == "webcam" and self.cap is None:
            return
        self._processing = True
        idx = self.current_index
        img_path = self.image_files[idx] if self.source_mode == "folder" else None

        def _infer():
            try:
                if self.source_mode == "webcam":
                    ok, frame = self.cap.read()
                    if not ok:
                        raise RuntimeError("Unable to read frame from webcam")
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frame_name = f"CAM {self.camera_index}"
                else:
                    # Use pre-loaded image if available, else load from disk
                    if idx in self._preload_buffer:
                        pil_img = self._preload_buffer.pop(idx)
                    else:
                        pil_img = Image.open(img_path).convert("L").convert("RGB")
                    frame_name = img_path.name

                # Time ONLY inference (not disk I/O)
                t0 = time.time()
                results = self.model.predict(
                    source=pil_img, conf=self.confidence, imgsz=640,
                    verbose=False, half=True,
                )
                infer_ms = (time.time() - t0) * 1000

                # Build annotated frame
                annotated_bgr = results[0].plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]
                ann_pil = Image.fromarray(annotated_rgb)

                # Collect detections
                dets = []
                result = results[0]
                obb = result.obb
                if obb is not None and len(obb):
                    for i in range(len(obb)):
                        cls_id = int(obb.cls[i])
                        cls_name = result.names[cls_id]
                        conf = float(obb.conf[i])
                        dets.append({"class": cls_name, "conf": conf})
                elif result.boxes is not None and len(result.boxes):
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = result.names[cls_id]
                        conf = float(box.conf[0])
                        dets.append({"class": cls_name, "conf": conf})

                # Pre-load upcoming images while GPU was working
                if self.source_mode == "folder":
                    self._preload_upcoming(idx)

                # Reference cross-check (US / Czech) for low-confidence detections
                ref_notes: list = []
                if self.ref_enabled and self.ref_indices and dets:
                    for d in dets:
                        if d["conf"] < self.ref_fallback_conf:
                            d_hint = CANONICAL_CONNECTORS.get(d["class"], d["class"])
                            if d_hint not in _D_CANONICAL:
                                d_hint = d["class"] if d["class"] in _D_CANONICAL else ""
                            if d_hint:
                                ref_cls, ref_sim = ref_best_match(pil_img, d_hint, self.ref_indices)
                                if ref_cls != d_hint and ref_sim > 0.45:
                                    ref_notes.append(
                                        f"  \u21b3 ref: {d_hint}\u2192{ref_cls} sim={ref_sim:.3f} [override]"
                                    )
                                else:
                                    ref_notes.append(
                                        f"  \u21b3 ref: {d_hint} confirmed sim={ref_sim:.3f}"
                                    )

                # Schedule UI update on main thread
                self.root.after(0, lambda rn=ref_notes: self._update_ui(
                    ann_pil, dets, frame_name, infer_ms, pil_img, auto_advance, rn
                ))

            except Exception as e:
                failed_name = img_path.name if img_path else f"CAM {self.camera_index}"
                self.root.after(0, lambda: self._log_det(f"✗ {failed_name}: {e}\n"))
                self.root.after(0, lambda: self._finish_frame(auto_advance))

        threading.Thread(target=_infer, daemon=True).start()

    def _preload_upcoming(self, current_idx: int):
        """Pre-load next N images into memory so disk I/O never blocks inference."""
        for offset in range(1, self._preload_ahead + 1):
            nxt = (current_idx + offset) % len(self.image_files)
            if nxt not in self._preload_buffer:
                try:
                    path = self.image_files[nxt]
                    self._preload_buffer[nxt] = (
                        Image.open(path).convert("L").convert("RGB")
                    )
                except Exception:
                    pass
        # Evict stale entries
        keep = set(
            (current_idx + i) % len(self.image_files)
            for i in range(self._preload_ahead + 1)
        )
        for k in list(self._preload_buffer):
            if k not in keep:
                del self._preload_buffer[k]

    def _finish_frame(self, auto_advance: bool):
        """Called after frame processing completes — advances index if playing."""
        self._processing = False
        if auto_advance:
            if self.source_mode == "folder" and self.image_files:
                self.current_index = (self.current_index + 1) % len(self.image_files)
                self.image_listbox.selection_clear(0, tk.END)
                self.image_listbox.selection_set(self.current_index)
                self.image_listbox.see(self.current_index)
            if self.playing:
                self.root.after(self.frame_delay_ms, self._play_loop)

    # ── UI updates (all on main thread) ───────────────────────────────────
    def _update_ui(self, ann_pil: Image.Image, dets: list, filename: str,
                   elapsed_ms: float, original_pil: Image.Image,
                   auto_advance: bool = False, ref_notes: list | None = None):
        # --- Canvas ---
        self._display_frame(ann_pil)
        self.cam_file.config(text=filename)
        self.cam_fps.config(text=f"{elapsed_ms:.0f} ms")
        self.canvas.delete("placeholder")

        # --- Detection log ---
        self.total_frames += 1
        ts = datetime.now().strftime("%H:%M:%S")
        if dets:
            self.total_detections += len(dets)
            for d in dets:
                sev = severity_label(d["conf"])
                self._log_det(
                    f"[{ts}] {filename} | {d['class']}"
                    f" ({CANONICAL_CONNECTORS.get(d['class'], d['class'])}) "
                    f"({d['conf']:.0%}) [{sev}]\n"
                )
                self.class_counts[d["class"]] = self.class_counts.get(d["class"], 0) + 1
            if ref_notes:
                for note in ref_notes:
                    self._log_det(note + "\n")
        else:
            self._log_det(f"[{ts}] {filename} | No damage detected\n")

        # --- Class breakdown ---
        self._update_class_breakdown(dets)

        # --- Stats ---
        elapsed_s = int(time.time() - self.start_time) if self.start_time else 0
        mins, secs = divmod(elapsed_s, 60)
        self.stats_label.config(
            text=f"Frames: {self.total_frames}\n"
                 f"Detections: {self.total_detections}\n"
                 f"Elapsed: {mins}m {secs}s"
        )
        self._draw_histogram()

        # --- AI analysis (async, only when detections found) ---
        if dets and REQUESTS_AVAILABLE:
            if self.openai_api_key:
                threading.Thread(target=self._ask_openai, daemon=True,
                                 args=(dets, filename, original_pil)).start()
            else:
                threading.Thread(target=self._ask_ollama, daemon=True,
                                 args=(dets, filename, original_pil)).start()

        # --- Finish frame processing (advance if playing) ---
        self._finish_frame(auto_advance)

    def _display_frame(self, pil_img: Image.Image):
        """Fit annotated image into the canvas, preserving aspect ratio."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self._current_photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("frame")
        self.canvas.create_image(cw // 2, ch // 2, image=self._current_photo,
                                 anchor=tk.CENTER, tags="frame")

    def _on_canvas_resize(self, _event):
        if self._current_photo:
            # Re-display will happen on next frame; avoid flicker
            pass

    def _release_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None

    def _on_close(self):
        self.playing = False
        self._release_camera()
        self.root.destroy()

    def _log_det(self, text: str):
        self.det_text.config(state=tk.NORMAL)
        self.det_text.insert(tk.END, text)
        self.det_text.see(tk.END)
        self.det_text.config(state=tk.DISABLED)

    def _update_class_breakdown(self, dets: list):
        self.class_text.config(state=tk.NORMAL)
        self.class_text.delete("1.0", tk.END)
        if not self.class_counts:
            self.class_text.insert(tk.END, "No detections yet.\n")
        else:
            for cls, count in sorted(self.class_counts.items()):
                desc = CLASS_DESCRIPTIONS.get(cls, cls)
                self.class_text.insert(tk.END, f"{cls} — {desc}\n")
                self.class_text.insert(tk.END, f"    Total: {count}\n\n")
            # Current frame detail
            if dets:
                self.class_text.insert(tk.END, "── Current Frame ──\n")
                for d in dets:
                    sev = severity_label(d["conf"])
                    self.class_text.insert(
                        tk.END,
                        f"  {d['class']} {d['conf']:.0%} [{sev}]\n"
                    )
        self.class_text.config(state=tk.DISABLED)

    def _draw_histogram(self):
        c = self.hist_canvas
        c.delete("all")
        if not self.class_counts:
            return
        w = c.winfo_width() or 280
        h = c.winfo_height() or 60
        max_count = max(self.class_counts.values()) or 1
        classes = sorted(self.class_counts.keys())
        bar_w = max(20, (w - 20) // max(len(classes), 1))
        x = 10
        for cls in classes:
            count = self.class_counts[cls]
            bar_h = int((count / max_count) * (h - 20))
            colour = CLASS_COLOURS.get(cls, CYAN)
            c.create_rectangle(x, h - 5 - bar_h, x + bar_w - 4, h - 5,
                               fill=colour, outline="")
            c.create_text(x + (bar_w - 4) // 2, h - 8 - bar_h, text=str(count),
                          fill=WHITE, font=("Arial", 7, "bold"), anchor=tk.S)
            c.create_text(x + (bar_w - 4) // 2, h - 2, text=cls,
                          fill=GREY, font=("Arial", 6), anchor=tk.S)
            x += bar_w

    # ── Ollama interpretation  ────────────────────────────────────────────
    def _resolve_ollama_model(self) -> str:
        """Return a working model name, preferring the configured one."""
        configured = (self.ollama_model or "").strip()
        if not configured:
            return "qwen2.5vl:7b"

        # If already tagged (e.g. qwen2.5vl:7b), use it directly.
        if ":" in configured:
            return configured

        # Try to discover a matching tagged model from the local Ollama registry.
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            names = [str(m.get("name", "")).strip() for m in models]
            prefixed = [n for n in names if n.startswith(configured + ":")]
            if prefixed:
                return sorted(prefixed)[0]
        except Exception:
            pass

        return configured

    def _ask_ollama(self, dets: list, filename: str, pil_img: Image.Image):
        try:
            det_text = "\n".join(
                f"- {d['class']} ({CLASS_DESCRIPTIONS.get(d['class'], '')}) "
                f"conf={d['conf']:.2f}" for d in dets
            )
            prompt = (
                f"Road image '{filename}' was analysed by a YOLO road damage model.\n"
                f"Detections:\n{det_text}\n\n"
                "Give a brief 2-sentence road condition assessment and maintenance recommendation."
            )

            # Encode image for vision model
            buf = io.BytesIO()
            small = pil_img.copy()
            small.thumbnail((768, 768), Image.LANCZOS)
            small.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            model_name = self._resolve_ollama_model()
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }

            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
            )

            # Some installs return 404 when model name is missing a required tag.
            if resp.status_code == 404 and ":" not in model_name:
                payload["model"] = self._resolve_ollama_model()
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60,
                )

            resp.raise_for_status()
            answer = resp.json().get("response", "")
            self.root.after(0, lambda: self._show_analysis(answer, filename, "Ollama"))
        except Exception as e:
            self.root.after(0, lambda: self._show_analysis(
                f"Ollama unavailable: {e}", filename, "Ollama"))

    def _ask_openai(self, dets: list, filename: str, pil_img: Image.Image):
        try:
            det_text = "\n".join(
                f"- {d['class']} ({CLASS_DESCRIPTIONS.get(d['class'], '')}) "
                f"conf={d['conf']:.2f}" for d in dets
            )
            prompt = (
                f"Road image '{filename}' was analysed by a YOLO road damage model.\n"
                f"Detections:\n{det_text}\n\n"
                "Give a brief 2-sentence road condition assessment and maintenance recommendation."
            )

            buf = io.BytesIO()
            small = pil_img.copy()
            small.thumbnail((768, 768), Image.LANCZOS)
            small.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                },
                timeout=30,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            self.root.after(0, lambda: self._show_analysis(answer, filename, "GPT-4o-mini"))
        except Exception as e:
            self.root.after(0, lambda: self._show_analysis(
                f"OpenAI error: {e}", filename, "GPT-4o-mini"))

    def _show_analysis(self, text: str, filename: str, source: str = "Ollama"):
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete("1.0", tk.END)
        ts = datetime.now().strftime("%H:%M:%S")
        self.analysis_text.insert(tk.END, f"[{ts}] [{source}] {filename}\n\n{text}\n")
        self.analysis_text.config(state=tk.DISABLED)

    # ── Run ───────────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = LiveCamApp()
    app.run()
