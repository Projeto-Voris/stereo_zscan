#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_zscan_unified.py  -  Stereo Z-Scan  |  Unified Execution + Debug Tool

Left panel  : parameter configuration tabs + Run / Stop controls
Right panel : debug/diagnostic visualization tabs + execution log

Changes (v2):
  - Direct images_folder input (replaces objects/distances loop)
  - process_coarse_to_fine (parabolic sub-voxel) toggle in Grade 3D tab
  - Runner generates single-folder C2F or brute-force pipeline
"""

import sys, os, json, threading, subprocess, time
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from include.SpatioCorrelation import PyTorchStereoCorrel
    HAS_ZSCAN = True
except ImportError as _e:
    HAS_ZSCAN = False

CONFIG_FILE = Path(__file__).parent / ".gui_unified_config.json"
_DEFAULT_PYTHON = (str(Path(__file__).parent / ".venv/bin/python")
                   if (Path(__file__).parent / ".venv/bin/python").exists()
                   else sys.executable)

DEFAULT_CONFIG: dict = {
    # ── Execution ──────────────────────────────────────────────────────────────
    "python_path":    _DEFAULT_PYTHON,
    "yaml_file":      "/home/daniel/Pictures/20260928 - Avaliação reconstrução sistemas/active_stereo/calib.yaml",
    "dataset_root":   "",   # root dir that contains timestamp subfolders
    "dataset_timestamps": "",  # comma-separated selected timestamps (empty = all)
    "black_factor": "1.0",
    "method_laser":  False,
    "method_fringe": True,
    "method_correl": False,
    "plot":           False,
    "debug":          False,
    "save_file":      True,
    "euclidian_filter": True,
    "mask_std":       True,
    "mask_uv":        True,
    # ── Grid / C2F ─────────────────────────────────────────────────────────────
    "use_c2f":        True,        # True = process_coarse_to_fine, False = brute-force
    "g1_xy":          "2.0",       # brute-force pass-1 XY step  /  C2F fine XY step
    "g1_z":           "0.5",       # brute-force pass-1 Z step   /  C2F fine Z step
    "g2_xy":          "1.0",       # brute-force pass-2 XY step  (unused in C2F)
    "g2_z":           "0.1",       # brute-force pass-2 Z step   (unused in C2F)
    "dz_coarse":      "2.0",       # C2F coarse Z step [mm]
    "fine_window":    "3.0",       # C2F fine half-window [mm]
    "surface_following": True,     # C2F slope-corrected patches
    "diag_x_min": "0",   "diag_x_max": "500",
    "diag_y_min": "0",   "diag_y_max": "400",
    "diag_z_min": "-100","diag_z_max": "100",
    "proc_x_min": "-100","proc_x_max": "500",
    "proc_y_min": "-100","proc_y_max": "400",
    # ── Per-method ─────────────────────────────────────────────────────────────
    "laser_n_imgs": "5",  "laser_kernels": "3",
    "laser_thresh": "0.7","laser_radius": "10.0",
    "laser_min_neighbors": "5","laser_mask_bounds": "20",
    "correl_n_imgs": "15","correl_kernels": "1",
    "correl_thresh": "0.9","correl_radius": "15.0",
    "correl_min_neighbors": "5","correl_mask_bounds": "30",
    "fringe_n_imgs": "1", "fringe_kernels": "1",
    "fringe_px_f":  "64", "fringe_steps": "8",
    "fringe_thresh": "0.1","fringe_radius": "10.0",
    "fringe_min_neighbors": "15","fringe_mask_bounds": "30",
    # ── Debug shared ───────────────────────────────────────────────────────────
    "method":         "fringe",
    "n_images":       16,
    "apply_clahe":    True,
    "undistort":      True,
    "clahe_clip":     2.0,
    "sat_threshold":  245,
    "cam_view":       "left",
    "proj_source":    "grid",
    "proj_color":     "Z",
    "cloud_csv":      "",
    "filt_threshold": 0.1,
    "filt_radius":    10.0,
    "filt_min_neighbors": 5,
    "filt_mask_bounds": 10.0,
    "filt_use_uv":    True,
    "filt_use_std":   True,
    "filt_use_euclidean": True,
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            conf = DEFAULT_CONFIG.copy(); conf.update(saved); return conf
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(conf: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(conf, indent=2))
    except Exception as e:
        print(f"save_config error: {e}")


def _dark(fig, axes):
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#2a2a3e"); ax.tick_params(colors="#cccccc")
        ax.title.set_color("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.xaxis.label.set_color("#cccccc"); ax.yaxis.label.set_color("#cccccc")


# ══════════════════════════════════════════════════════════════════════════════
class UnifiedApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stereo Z-Scan  |  Unified Control & Debug  v2")
        self.root.geometry("1750x1000"); self.root.minsize(1200, 700)
        self.cfg = load_config()
        self.process = None; self.worker_thread = None
        self.zscan: "PyTorchStereoCorrel | None" = None
        self.left_imgs_raw: list = []; self.right_imgs_raw: list = []
        self.left_imgs:  list = [];    self.right_imgs: list  = []
        self.fringe_abs_left  = None;  self.fringe_abs_right = None
        self.fringe_mod_left  = None;  self.fringe_mod_right = None
        self.xyz_cloud = None;         self.corr_cloud = None
        self.roi = None
        self._roi_dragging = False; self._press_xy = None
        self._roi_patch = None;     self._proj_cbar = None
        self._filter_debounce_id = None
        self._filter_busy = False;  self._filter_pending = False
        self._init_vars()
        self._build_ui()

    # ── Variables ─────────────────────────────────────────────────────────────
    def _init_vars(self) -> None:
        C = self.cfg
        self.status_var = tk.StringVar(value="Pronto.")

        self.runner_vars: dict = {}
        for k, default in DEFAULT_CONFIG.items():
            if isinstance(default, bool):
                self.runner_vars[k] = tk.BooleanVar(value=bool(C.get(k, default)))
            elif isinstance(default, float):
                self.runner_vars[k] = tk.DoubleVar(value=float(C.get(k, default)))
            elif isinstance(default, int) and not isinstance(default, bool):
                self.runner_vars[k] = tk.IntVar(value=int(C.get(k, default)))
            else:
                self.runner_vars[k] = tk.StringVar(value=str(C.get(k, default)))

        # Aliases for shared references
        self.yaml_var          = self.runner_vars["yaml_file"]
        self.dataset_root_var  = self.runner_vars["dataset_root"]
        self.gx0               = self.runner_vars["diag_x_min"]
        self.gx1               = self.runner_vars["diag_x_max"]
        self.gy0               = self.runner_vars["diag_y_min"]
        self.gy1               = self.runner_vars["diag_y_max"]
        self.gz0               = self.runner_vars["diag_z_min"]
        self.gz1               = self.runner_vars["diag_z_max"]
        self.gxy_step          = self.runner_vars["g1_xy"]
        self.gz_step           = self.runner_vars["g1_z"]
        self.method_var        = self.runner_vars["method"]
        self.n_imgs_var        = self.runner_vars["n_images"]
        self.clahe_var         = self.runner_vars["apply_clahe"]
        self.undist_var        = self.runner_vars["undistort"]
        self.clahe_clip_var    = self.runner_vars["clahe_clip"]
        self.fringe_pxf_var    = self.runner_vars["fringe_px_f"]
        self.fringe_steps_var  = self.runner_vars["fringe_steps"]
        self.sat_thresh_var    = self.runner_vars["sat_threshold"]
        self.cam_view_var      = self.runner_vars["cam_view"]
        self.proj_source_var   = self.runner_vars["proj_source"]
        self.proj_color_var    = self.runner_vars["proj_color"]
        self.cloud_csv_var     = self.runner_vars["cloud_csv"]
        self.filt_thresh_var   = self.runner_vars["filt_threshold"]
        self.filt_radius_var   = self.runner_vars["filt_radius"]
        self.filt_minn_var     = self.runner_vars["filt_min_neighbors"]
        self.filt_bounds_var   = self.runner_vars["filt_mask_bounds"]
        self.filt_uv_var       = self.runner_vars["filt_use_uv"]
        self.filt_std_var      = self.runner_vars["filt_use_std"]
        self.filt_eucl_var     = self.runner_vars["filt_use_euclidean"]

        # UI-only state
        self.roi_x0_var    = tk.StringVar(value="0")
        self.roi_y0_var    = tk.StringVar(value="0")
        self.roi_x1_var    = tk.StringVar(value="100")
        self.roi_y1_var    = tk.StringVar(value="100")
        self.img_idx_var   = tk.IntVar(value=0)
        self.proj_img_idx  = tk.IntVar(value=0)
        self.filt_counts_var = tk.StringVar(value="N/A")
        self.diff_col_var  = tk.IntVar(value=0)
        self.diff_line_var = tk.IntVar(value=0)
        # Timestamp selector (UI-only, populated when root is browsed)
        self._timestamps: list = []          # full list of found timestamps
        self.ts_selected_var = tk.StringVar(value="")  # currently active timestamp for debug

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook.Tab", padding=[10, 4])
        style.configure("Hdr.TLabel", font=("Helvetica", 9, "bold"))

        top = ttk.Frame(self.root, padding=(6, 3)); top.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(top, text="Stereo Z-Scan  |  Unified Control & Debug  v2",
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(0, 16))
        self.btn_run  = ttk.Button(top, text="▶ Executar",  command=self._start_execution)
        self.btn_stop = ttk.Button(top, text="⏹ Parar",    command=self._stop_execution, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=3)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="💾 Salvar Config", command=self._save_cfg).pack(side=tk.LEFT, padx=3)
        self.status_lbl = ttk.Label(top, textvariable=self.status_var,
                                    font=("Helvetica", 9, "bold"), foreground="#0066cc")
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        sb = ttk.Frame(self.root, relief="sunken", padding=2); sb.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(sb, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, padx=4)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        lf = ttk.Frame(paned, width=500); paned.add(lf, weight=0)
        rf = ttk.Frame(paned);            paned.add(rf, weight=1)
        self._build_param_panel(lf)
        self._build_debug_panel(rf)

    # ══════════════════════════════════════════════════════════════════════════
    # Left panel
    # ══════════════════════════════════════════════════════════════════════════
    def _build_param_panel(self, parent) -> None:
        nb = ttk.Notebook(parent); nb.pack(fill=tk.BOTH, expand=True)
        tabs = {}
        for name in ["Geral", "Grade 3D / C2F", "Laser", "Fringe", "Correl"]:
            t = ttk.Frame(nb, padding=8); nb.add(t, text=f"  {name}  "); tabs[name] = t
        self._build_general_tab(tabs["Geral"])
        self._build_grid_tab(tabs["Grade 3D / C2F"])
        self._build_method_fields(tabs["Laser"],  "laser",  False)
        self._build_method_fields(tabs["Fringe"], "fringe", True)
        self._build_method_fields(tabs["Correl"], "correl", False)

    # ── Geral tab ─────────────────────────────────────────────────────────────
    def _build_general_tab(self, parent) -> None:
        cv = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=cv.yview)
        sf = ttk.Frame(cv)
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw")
        cv.configure(yscrollcommand=vsb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        row = [0]

        def _lbl(t):
            ttk.Label(sf, text=t, style="Hdr.TLabel").grid(
                row=row[0], column=0, sticky=tk.W, pady=2); row[0] += 1

        def _sep():
            ttk.Separator(sf, orient=tk.HORIZONTAL).grid(
                row=row[0], column=0, columnspan=2, sticky=tk.EW, pady=4); row[0] += 1

        def _e(label, var, browse=None):
            ttk.Label(sf, text=label).grid(row=row[0], column=0, sticky=tk.W, pady=1)
            if browse:
                f = ttk.Frame(sf); f.grid(row=row[0], column=1, sticky=tk.EW, pady=1)
                ttk.Entry(f, textvariable=var, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
                ttk.Button(f, text="...", width=3, command=browse).pack(side=tk.LEFT)
            else:
                ttk.Entry(sf, textvariable=var, width=26).grid(
                    row=row[0], column=1, sticky=tk.EW, pady=1)
            row[0] += 1

        _lbl("Arquivo YAML (Calibracao):")
        _e("YAML:", self.yaml_var, self._browse_yaml)
        _lbl("Interpretador Python:")
        _e("Python:", self.runner_vars["python_path"], self._browse_python)
        _sep()

        _lbl("Dataset Root (contém pastas de timestamp):")
        ttk.Label(sf, text="Root:").grid(row=row[0], column=0, sticky=tk.W, pady=1)
        rf_ = ttk.Frame(sf); rf_.grid(row=row[0], column=1, sticky=tk.EW, pady=1); row[0] += 1
        self._geral_root_entry = ttk.Entry(rf_, textvariable=self.dataset_root_var, width=22)
        self._geral_root_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rf_, text="...", width=3, command=self._browse_dataset_root).pack(side=tk.LEFT)

        # Timestamps multi-select listbox
        ttk.Label(sf, text="Timestamps:").grid(row=row[0], column=0, sticky=tk.NW, pady=1)
        ts_frame = ttk.Frame(sf); ts_frame.grid(row=row[0], column=1, sticky=tk.EW, pady=1); row[0] += 1
        ts_scroll = ttk.Scrollbar(ts_frame, orient=tk.VERTICAL)
        self.ts_listbox = tk.Listbox(ts_frame, selectmode=tk.EXTENDED, height=6,
                                     yscrollcommand=ts_scroll.set,
                                     bg="#1e1e2e", fg="#e0e0e0",
                                     selectbackground="#3355aa",
                                     font=("Consolas", 9))
        ts_scroll.config(command=self.ts_listbox.yview)
        self.ts_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ts_scroll.pack(side=tk.LEFT, fill=tk.Y)

        ts_btn_frame = ttk.Frame(sf)
        ts_btn_frame.grid(row=row[0], column=1, sticky=tk.W, pady=2); row[0] += 1
        ttk.Button(ts_btn_frame, text="Todos",  width=6, command=self._ts_select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(ts_btn_frame, text="Nenhum", width=6, command=self._ts_select_none).pack(side=tk.LEFT, padx=2)
        ttk.Button(ts_btn_frame, text="⟳ Ler",  width=6, command=self._scan_timestamps).pack(side=tk.LEFT, padx=2)

        self.ts_info_lbl = ttk.Label(sf, text="—", foreground="gray", wraplength=230)
        self.ts_info_lbl.grid(row=row[0], column=1, sticky=tk.W); row[0] += 1

        _e("Black Factor:", self.runner_vars["black_factor"])
        _sep()

        _lbl("Metodos a executar:")
        mf = ttk.Frame(sf); mf.grid(row=row[0], column=0, columnspan=2, sticky=tk.W); row[0] += 1
        for m in ["laser", "fringe", "correl"]:
            ttk.Checkbutton(mf, text=m.capitalize(),
                            variable=self.runner_vars[f"method_{m}"]).pack(side=tk.LEFT, padx=4)
        _sep()

        _lbl("Flags:")
        for k, t in [("plot",             "Plotar Nuvem 3D"),
                     ("debug",            "Debug Intermediario"),
                     ("save_file",        "Salvar CSV"),
                     ("euclidian_filter", "Filtro Euclidiano"),
                     ("mask_std",         "Mascara STD/Mod"),
                     ("mask_uv",          "Mascara UV")]:
            ttk.Checkbutton(sf, text=t, variable=self.runner_vars[k]).grid(
                row=row[0], column=0, columnspan=2, sticky=tk.W, pady=1); row[0] += 1

    # ── Grade 3D / C2F tab ────────────────────────────────────────────────────
    def _build_grid_tab(self, parent) -> None:
        row = [0]

        def _rng(lbl, v0, v1):
            ttk.Label(parent, text=lbl).grid(row=row[0], column=0, sticky=tk.W, pady=1)
            f = ttk.Frame(parent); f.grid(row=row[0], column=1, sticky=tk.W, pady=1)
            ttk.Entry(f, textvariable=v0, width=7).pack(side=tk.LEFT)
            ttk.Label(f, text=" a ").pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=v1, width=7).pack(side=tk.LEFT); row[0] += 1

        def _e(lbl, var, w=10):
            ttk.Label(parent, text=lbl).grid(row=row[0], column=0, sticky=tk.W, pady=1)
            ttk.Entry(parent, textvariable=var, width=w).grid(
                row=row[0], column=1, sticky=tk.W); row[0] += 1

        def _sep():
            ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
                row=row[0], column=0, columnspan=2, sticky=tk.EW, pady=6); row[0] += 1

        # ── Limites da Grade ──────────────────────────────────────────────────
        ttk.Label(parent, text="Limites da Grade 3D (mm)", style="Hdr.TLabel").grid(
            row=row[0], column=0, columnspan=2, sticky=tk.W, pady=(0, 4)); row[0] += 1
        _rng("X (mm):", self.gx0, self.gx1)
        _rng("Y (mm):", self.gy0, self.gy1)
        _rng("Z (mm):", self.gz0, self.gz1)
        _sep()

        # ── Modo de Processamento ─────────────────────────────────────────────
        ttk.Label(parent, text="Modo de Processamento", style="Hdr.TLabel").grid(
            row=row[0], column=0, columnspan=2, sticky=tk.W, pady=(0, 4)); row[0] += 1

        bf_frame = ttk.Frame(parent)
        bf_frame.grid(row=row[0], column=0, columnspan=2, sticky=tk.W, pady=2); row[0] += 1
        ttk.Radiobutton(bf_frame, text="Força Bruta (2-pass process_segmented_z)",
                        variable=self.runner_vars["use_c2f"], value=False,
                        command=self._on_c2f_toggle).pack(anchor=tk.W)
        ttk.Radiobutton(bf_frame, text="Coarse-to-Fine + Parabólico (process_coarse_to_fine)",
                        variable=self.runner_vars["use_c2f"], value=True,
                        command=self._on_c2f_toggle).pack(anchor=tk.W)

        _sep()

        # ── Brute-force params ────────────────────────────────────────────────
        self._bf_lbl = ttk.Label(parent, text="Força Bruta — Passos", style="Hdr.TLabel")
        self._bf_lbl.grid(row=row[0], column=0, columnspan=2, sticky=tk.W, pady=(0, 4)); row[0] += 1
        self._bf_widgets = []

        def _bfe(lbl, var, w=10):
            l = ttk.Label(parent, text=lbl); l.grid(row=row[0], column=0, sticky=tk.W, pady=1)
            e = ttk.Entry(parent, textvariable=var, width=w)
            e.grid(row=row[0], column=1, sticky=tk.W); row[0] += 1
            self._bf_widgets.extend([l, e])

        _bfe("Passo XY 1a iter. (mm):", self.runner_vars["g1_xy"])
        _bfe("Passo Z  1a iter. (mm):", self.runner_vars["g1_z"])
        _bfe("Passo XY 2a iter. (mm):", self.runner_vars["g2_xy"])
        _bfe("Passo Z  2a iter. (mm):", self.runner_vars["g2_z"])

        # ── C2F params ────────────────────────────────────────────────────────
        self._c2f_lbl = ttk.Label(parent, text="Coarse-to-Fine — Parâmetros", style="Hdr.TLabel")
        self._c2f_lbl.grid(row=row[0], column=0, columnspan=2, sticky=tk.W, pady=(0, 4)); row[0] += 1
        self._c2f_widgets = [self._c2f_lbl]

        def _c2fe(lbl, var, w=10):
            l = ttk.Label(parent, text=lbl); l.grid(row=row[0], column=0, sticky=tk.W, pady=1)
            e = ttk.Entry(parent, textvariable=var, width=w)
            e.grid(row=row[0], column=1, sticky=tk.W); row[0] += 1
            self._c2f_widgets.extend([l, e])

        _c2fe("Passo XY (mm)  [grade fina]:", self.runner_vars["g1_xy"])
        _c2fe("Passo Z fino (mm):",            self.runner_vars["g1_z"])
        _c2fe("dz Coarse (mm):",               self.runner_vars["dz_coarse"])
        _c2fe("Janela Fina ±(mm):",            self.runner_vars["fine_window"])
        l = ttk.Label(parent, text="Surface Following:")
        l.grid(row=row[0], column=0, sticky=tk.W, pady=1)
        cb = ttk.Checkbutton(parent, variable=self.runner_vars["surface_following"])
        cb.grid(row=row[0], column=1, sticky=tk.W); row[0] += 1
        self._c2f_widgets.extend([l, cb])

        _sep()

        # ── Limits XY Processamento ───────────────────────────────────────────
        ttk.Label(parent, text="Limites XY Processamento", style="Hdr.TLabel").grid(
            row=row[0], column=0, columnspan=2, sticky=tk.W, pady=(0, 4)); row[0] += 1
        for ax in ["x", "y"]:
            ttk.Label(parent, text=f"Proc {ax.upper()} (Min,Max):").grid(
                row=row[0], column=0, sticky=tk.W, pady=1)
            f = ttk.Frame(parent); f.grid(row=row[0], column=1, sticky=tk.W, pady=1)
            ttk.Entry(f, textvariable=self.runner_vars[f"proc_{ax}_min"], width=7).pack(side=tk.LEFT)
            ttk.Label(f, text=" a ").pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=self.runner_vars[f"proc_{ax}_max"], width=7).pack(side=tk.LEFT)
            row[0] += 1

        # Apply initial C2F toggle state
        self._on_c2f_toggle()

    def _on_c2f_toggle(self):
        """Show/hide brute-force vs C2F parameter widgets."""
        use_c2f = self.runner_vars["use_c2f"].get()
        state_bf  = tk.DISABLED if use_c2f  else tk.NORMAL
        state_c2f = tk.NORMAL   if use_c2f  else tk.DISABLED
        for w in self._bf_widgets:
            try: w.config(state=state_bf)
            except Exception: pass
        for w in self._c2f_widgets:
            try: w.config(state=state_c2f)
            except Exception: pass

    # ── Per-method fields ─────────────────────────────────────────────────────
    def _build_method_fields(self, parent, prefix, is_fringe) -> None:
        row = [0]

        def _e(lbl, k):
            ttk.Label(parent, text=lbl).grid(row=row[0], column=0, sticky=tk.W, pady=1)
            ttk.Entry(parent, textvariable=self.runner_vars[k], width=18).grid(
                row=row[0], column=1, sticky=tk.W, pady=1); row[0] += 1

        ttk.Label(parent, text=f"Parametros: {prefix.capitalize()}",
                  style="Hdr.TLabel").grid(row=row[0], column=0, columnspan=2,
                                           sticky=tk.W, pady=(0, 6)); row[0] += 1
        _e("N Imagens (virgula):", f"{prefix}_n_imgs")
        _e("Kernels (virgula):",   f"{prefix}_kernels")
        if is_fringe:
            _e("Pixels/fringe:", f"{prefix}_px_f")
            _e("Steps:",         f"{prefix}_steps")
        _e("Threshold (<thr):" if is_fringe else "Threshold (>thr):", f"{prefix}_thresh")
        _e("Raio Euclidiano:",   f"{prefix}_radius")
        _e("Min. Vizinhos:",     f"{prefix}_min_neighbors")
        _e("Bounds Mod:" if is_fringe else "Bounds STD:", f"{prefix}_mask_bounds")

    # ══════════════════════════════════════════════════════════════════════════
    # Right panel
    # ══════════════════════════════════════════════════════════════════════════
    def _build_debug_panel(self, parent) -> None:
        nb = ttk.Notebook(parent); nb.pack(fill=tk.BOTH, expand=True)
        tabs = {}
        for name in ["Imagens", "ROI & Analise", "Projecao 3D", "Filtros", "Log"]:
            t = ttk.Frame(nb, padding=4); nb.add(t, text=f"  {name}  "); tabs[name] = t
        self._build_images_tab(tabs["Imagens"])
        self._build_roi_tab(tabs["ROI & Analise"])
        self._build_projection_tab(tabs["Projecao 3D"])
        self._build_filter_tab(tabs["Filtros"])
        self._build_log_tab(tabs["Log"])

    # ── Tab: Images ───────────────────────────────────────────────────────────
    def _build_images_tab(self, parent) -> None:
        ctrl = ttk.Frame(parent, width=310); ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        ctrl.pack_propagate(False)
        prev = ttk.Frame(parent); prev.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r = [0]

        def _g(w, c=0, cs=1, **kw): w.grid(row=r[0], column=c, columnspan=cs, **kw)
        def _lbl(t): _g(ttk.Label(ctrl, text=t, style="Hdr.TLabel"), sticky="w"); r[0] += 1
        def _sep(): _g(ttk.Separator(ctrl, orient="horizontal"), cs=3, sticky="ew", pady=4); r[0] += 1

        _lbl("YAML:")
        yf = ttk.Frame(ctrl); _g(yf, cs=3, sticky="ew"); r[0] += 1
        ttk.Entry(yf, textvariable=self.yaml_var, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(yf, text="...", width=3, command=self._browse_yaml).pack(side=tk.LEFT)
        _sep()

        _lbl("Metodo (subpasta):")
        for m in ["laser", "fringe", "correl"]:
            _g(ttk.Radiobutton(ctrl, text=m, variable=self.method_var, value=m,
                               command=self._update_roi_image), cs=3, sticky="w"); r[0] += 1
        _sep()

        _lbl("Dataset Root:")
        df2 = ttk.Frame(ctrl); _g(df2, cs=3, sticky="ew"); r[0] += 1
        ttk.Entry(df2, textvariable=self.dataset_root_var, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(df2, text="...", width=3,
                   command=self._browse_dataset_root).pack(side=tk.LEFT)

        _lbl("Timestamp:")
        ts_dbg_frame = ttk.Frame(ctrl); _g(ts_dbg_frame, cs=3, sticky="ew"); r[0] += 1
        self.ts_dbg_combo = ttk.Combobox(ts_dbg_frame, textvariable=self.ts_selected_var,
                                          width=18, state="readonly")
        self.ts_dbg_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ts_dbg_frame, text="⟳", width=2,
                   command=self._scan_timestamps).pack(side=tk.LEFT, padx=2)

        for lbl, var, w in [("N Imagens:",    self.n_imgs_var,      8),
                             ("CLAHE Clip:",   self.clahe_clip_var,  8),
                             ("px_f:",         self.fringe_pxf_var,  8),
                             ("Steps:",        self.fringe_steps_var,8),
                             ("Sat. Thr:",     self.sat_thresh_var,  8)]:
            ttk.Label(ctrl, text=lbl).grid(row=r[0], column=0, sticky="w")
            ttk.Entry(ctrl, textvariable=var, width=w).grid(row=r[0], column=1, sticky="w"); r[0] += 1

        _g(ttk.Checkbutton(ctrl, text="CLAHE",     variable=self.clahe_var),  cs=3, sticky="w"); r[0] += 1
        _g(ttk.Checkbutton(ctrl, text="Undistort", variable=self.undist_var), cs=3, sticky="w"); r[0] += 1

        self.btn_load = ttk.Button(ctrl, text="  Carregar Imagens  ", command=self._load_images)
        _g(self.btn_load, cs=3, sticky="ew", pady=4); r[0] += 1
        self.load_info = ttk.Label(ctrl, text="Nenhuma imagem.", foreground="gray", wraplength=290)
        _g(self.load_info, cs=3, sticky="w"); r[0] += 1
        _sep()

        _lbl("Preview:")
        for c in ["left", "right"]:
            _g(ttk.Radiobutton(ctrl, text=c, variable=self.cam_view_var, value=c,
                               command=self._update_preview_from_var), cs=3, sticky="w"); r[0] += 1
        self.prev_scale = ttk.Scale(ctrl, variable=self.img_idx_var, from_=0, to=0,
                                    orient=tk.HORIZONTAL, length=200,
                                    command=lambda v: self._update_preview(int(float(v))))
        _g(self.prev_scale, cs=3, sticky="ew"); r[0] += 1
        self.prev_lbl = ttk.Label(ctrl, text="img 0/0", foreground="gray")
        _g(self.prev_lbl, cs=3, sticky="w"); r[0] += 1

        self.prev_fig, self.prev_ax = plt.subplots(1, 1, figsize=(5, 4))
        _dark(self.prev_fig, [self.prev_ax])
        self.prev_canvas = FigureCanvasTkAgg(self.prev_fig, master=prev)
        self.prev_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.prev_canvas, prev).update()

    # ── Browse helpers ─────────────────────────────────────────────────────────
    def _browse_yaml(self):
        p = filedialog.askopenfilename(title="YAML", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
                                       initialdir=str(Path(__file__).parent / "cfg"))
        if p:
            try: p = os.path.relpath(p, Path(__file__).parent)
            except Exception: pass
            self.yaml_var.set(p)

    def _browse_python(self):
        p = filedialog.askopenfilename(title="Python executavel")
        if p: self.runner_vars["python_path"].set(p)

    def _browse_dataset_root(self):
        """Browse for the dataset root that contains timestamp subfolders."""
        cur = self.dataset_root_var.get().strip()
        initial = cur if cur and Path(cur).exists() else str(Path.home())
        d = filedialog.askdirectory(
            title="Pasta Root do Dataset (contém pastas timestamp/fringe, timestamp/laser ...)",
            initialdir=initial)
        if d:
            self.dataset_root_var.set(d)
            self.status_var.set(f"Root: {d}")
            self._scan_timestamps()

    def _scan_timestamps(self):
        """Scan the dataset root and populate the timestamp listbox + combobox."""
        root = self.dataset_root_var.get().strip()
        if not root or not Path(root).is_dir():
            self.ts_info_lbl.config(text="Root inválido.", foreground="orange")
            return
        # A valid timestamp dir must contain at least one of: fringe/, laser/, correl/
        methods = {"fringe", "laser", "correl"}
        ts_list = sorted(
            d.name for d in Path(root).iterdir()
            if d.is_dir() and any((d / m).is_dir() for m in methods)
        )
        self._timestamps = ts_list
        # Update Geral listbox
        if hasattr(self, "ts_listbox"):
            self.ts_listbox.delete(0, tk.END)
            for ts in ts_list:
                self.ts_listbox.insert(tk.END, ts)
            self.ts_listbox.select_set(0, tk.END)  # default: all selected
        # Update debug combobox
        if hasattr(self, "ts_dbg_combo"):
            self.ts_dbg_combo["values"] = ts_list
            if ts_list:
                self.ts_selected_var.set(ts_list[0])
        label = f"{len(ts_list)} timestamps encontrados."
        if hasattr(self, "ts_info_lbl"):
            self.ts_info_lbl.config(text=label, foreground="#88aaff")
        self.status_var.set(label)

    def _ts_select_all(self):
        if hasattr(self, "ts_listbox"):
            self.ts_listbox.select_set(0, tk.END)

    def _ts_select_none(self):
        if hasattr(self, "ts_listbox"):
            self.ts_listbox.selection_clear(0, tk.END)

    def _get_selected_timestamps(self) -> list:
        """Return the list of selected timestamps from the Geral listbox."""
        if not hasattr(self, "ts_listbox") or not self._timestamps:
            return self._timestamps
        sel_idx = self.ts_listbox.curselection()
        if not sel_idx:
            return self._timestamps  # empty selection = all
        return [self._timestamps[i] for i in sel_idx]

    def _browse_file(self, var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes,
                                       initialdir=str(Path(__file__).parent))
        if p:
            try: p = os.path.relpath(p, Path(__file__).parent)
            except Exception: pass
            var.set(p)

    def _resolve_debug_image_path(self):
        """Resolve (lp, rp) for the debug tab using root/timestamp/method/left + right."""
        root   = self.dataset_root_var.get().strip()
        ts     = self.ts_selected_var.get().strip()
        method = self.method_var.get().strip()
        if not root:
            raise RuntimeError("Selecione o Dataset Root.")
        if not ts:
            raise RuntimeError("Selecione um timestamp.")
        base = Path(root) / ts
        # method 'correl' uses the same image set as 'laser' if no correl subfolder
        method_dir = base / method
        if not method_dir.is_dir():
            raise RuntimeError(f"Subpasta não encontrada: {method_dir}")
        lp = method_dir / "left"
        rp = method_dir / "right"
        if not lp.is_dir() or not rp.is_dir():
            raise RuntimeError(f"Subpastas left/ e right/ não encontradas em {method_dir}")
        return lp, rp

    # ── Load images ────────────────────────────────────────────────────────────
    def _load_images(self):
        if not HAS_ZSCAN:
            messagebox.showerror("Erro", "PyTorchStereoCorrel nao disponivel."); return
        try:
            lp, rp = self._resolve_debug_image_path()
        except RuntimeError as e:
            messagebox.showwarning("Aviso", str(e)); return
        yaml = self.yaml_var.get().strip()
        if not yaml:
            messagebox.showwarning("Aviso", "Informe o YAML."); return
        self.btn_load.config(state=tk.DISABLED)
        self.load_info.config(text="Carregando...", foreground="orange")
        ts = self.ts_selected_var.get()
        method = self.method_var.get()
        self.status_var.set(f"Carregando  {ts}/{method} ...")

        def _w():
            try:
                base_gui = Path(__file__).parent
                lyaml = (base_gui / yaml) if not Path(yaml).is_absolute() else Path(yaml)
                n_max  = self._si(self.n_imgs_var.get(), 99)
                lf = sorted(lp.iterdir())[:n_max]
                rf = sorted(rp.iterdir())[:n_max]
                lr = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in lf]
                rr = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in rf]
                lr = [i for i in lr if i is not None]
                rr = [i for i in rr if i is not None]
                if not lr: raise RuntimeError(f"Sem imagens em {lp}")
                zscan = PyTorchStereoCorrel(yaml_file=str(lyaml))
                zscan.convert_images(lr, rr,
                                     apply_clahe=self.clahe_var.get(),
                                     undist=self.undist_var.get(),
                                     climp=self._sf(self.clahe_clip_var.get(), 2.0))
                lp2 = [zscan.left_images[i].cpu().numpy()  for i in range(len(lr))]
                rp2 = [zscan.right_images[i].cpu().numpy() for i in range(len(rr))]
                al = ar = ml = mr = None
                if method == "fringe":
                    try:
                        from include.fringe_projection.include.stereo_fringe_process import FringeProcess
                        px_f  = self._si(self.fringe_pxf_var.get(), 64)
                        steps = self._si(self.fringe_steps_var.get(), 8)
                        fp = FringeProcess(camera_resolution=lr[0].shape[::-1],
                                           px_f=px_f, steps=steps)
                        fp.images_left  = np.moveaxis(np.array(lr), 0, -1)
                        fp.images_right = np.moveaxis(np.array(rr), 0, -1)
                        a1, a2, m1, m2  = fp.calculate_abs_phi_images(visualize=False)
                        al = a1.cpu().numpy(); ar = a2.cpu().numpy()
                        ml = m1.cpu().numpy(); mr = m2.cpu().numpy()
                    except Exception as e:
                        print(f"[Fringe] {e}")
                self.root.after(0, self._on_imgs_loaded, zscan, lr, rr, lp2, rp2, al, ar, ml, mr, len(lr))
            except Exception as exc:
                self.root.after(0, self._on_load_err, str(exc))

        threading.Thread(target=_w, daemon=True).start()


    def _on_imgs_loaded(self, zscan, lr, rr, lp2, rp2, al, ar, ml, mr, n):
        self.zscan = zscan
        self.left_imgs_raw = lr; self.right_imgs_raw = rr
        self.left_imgs = lp2;   self.right_imgs = rp2
        self.fringe_abs_left = al; self.fringe_abs_right = ar
        self.fringe_mod_left = ml; self.fringe_mod_right = mr
        self.btn_load.config(state=tk.NORMAL)
        extra = " | Fase calc." if al is not None else ""
        self.load_info.config(text=f"{n} imgs [{self.method_var.get()}]{extra}",
                              foreground="#22cc66")
        self.status_var.set(f"Pronto. {n} imagens.")
        self.prev_scale.config(to=max(n - 1, 0)); self.img_idx_var.set(0)
        self._update_preview(0); self._update_roi_image()

    def _on_load_err(self, msg):
        self.btn_load.config(state=tk.NORMAL)
        self.status_var.set("Erro.")
        messagebox.showerror("Erro", msg)

    def _update_preview_from_var(self):
        self._update_preview(self.img_idx_var.get())

    def _update_preview(self, idx):
        if not self.left_imgs: return
        imgs = self.left_imgs if self.cam_view_var.get() == "left" else self.right_imgs
        idx = min(idx, len(imgs) - 1)
        self.prev_lbl.config(text=f"img {idx}/{len(imgs)-1}")
        self.prev_ax.clear()
        self.prev_ax.imshow(imgs[idx], cmap="gray", aspect="auto")
        self.prev_ax.set_title(f"{self.cam_view_var.get()} img {idx}", color="white")
        _dark(self.prev_fig, [self.prev_ax]); self.prev_ax.axis("off")
        self.prev_canvas.draw_idle()

    # ── Tab: ROI ──────────────────────────────────────────────────────────────
    def _build_roi_tab(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text=" Imagem (arrastar = ROI) ", padding=4)
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ctrl = ttk.Frame(lf); ctrl.pack(fill=tk.X, pady=2)
        ttk.Label(ctrl, text="Camera:").pack(side=tk.LEFT)
        for c in ["left", "right"]:
            ttk.Radiobutton(ctrl, text=c, variable=self.cam_view_var, value=c,
                            command=self._update_roi_image).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Limpar ROI", command=self._clear_roi).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="ROI=Img",   command=self._roi_full).pack(side=tk.LEFT)

        rc = ttk.Frame(lf); rc.pack(fill=tk.X, pady=2)
        for lbl, var in [("x0:", self.roi_x0_var), ("  y0:", self.roi_y0_var),
                         ("  x1:", self.roi_x1_var), ("  y1:", self.roi_y1_var)]:
            ttk.Label(rc, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(rc, textvariable=var, width=6).pack(side=tk.LEFT)
        ttk.Button(rc, text="Aplicar",     command=self._apply_manual_roi).pack(side=tk.LEFT, padx=4)
        ttk.Button(rc, text="Analisar ROI", command=self._run_roi_analysis).pack(side=tk.LEFT, padx=4)

        self.roi_fig, self.roi_ax = plt.subplots(1, 1, figsize=(5, 4))
        _dark(self.roi_fig, [self.roi_ax])
        self.roi_canvas = FigureCanvasTkAgg(self.roi_fig, master=lf)
        self.roi_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.roi_canvas, lf).update()
        self.roi_fig.canvas.mpl_connect("button_press_event",   self._roi_press)
        self.roi_fig.canvas.mpl_connect("motion_notify_event",  self._roi_motion)
        self.roi_fig.canvas.mpl_connect("button_release_event", self._roi_release)

        rf = ttk.Frame(parent, width=620)
        rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); rf.pack_propagate(False)
        rnb = ttk.Notebook(rf); rnb.pack(fill=tk.BOTH, expand=True)
        std_t  = ttk.Frame(rnb); ph_t  = ttk.Frame(rnb)
        mod_t  = ttk.Frame(rnb); diff_t = ttk.Frame(rnb); sat_t = ttk.Frame(rnb)
        rnb.add(std_t,  text="  STD  ")
        rnb.add(ph_t,   text="  Fase Abs.  ")
        rnb.add(mod_t,  text="  Modulacao  ")
        rnb.add(diff_t, text="  Diff Fase (3D)  ")
        rnb.add(sat_t,  text="  Saturacao  ")

        def _mc(fa, aa, ca, pt, info_a=None, it=""):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            _dark(fig, list(axes.ravel()))
            setattr(self, fa, fig); setattr(self, aa, axes)
            cnv = FigureCanvasTkAgg(fig, master=pt)
            cnv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            setattr(self, ca, cnv)
            if info_a:
                lbl = ttk.Label(pt, text=it, foreground="gray"); lbl.pack()
                setattr(self, info_a, lbl)

        _mc("std_fig",   "std_axes",   "std_canvas",   std_t,  "std_info",   "STD pixel-a-pixel.")
        _mc("phase_fig", "phase_axes", "phase_canvas", ph_t,   "phase_info", "Fase absoluta (rad).")
        _mc("mod_fig",   "mod_axes",   "mod_canvas",   mod_t,  "mod_info",   "Modulacao.")
        _mc("sat_fig",   "sat_axes",   "sat_canvas",   sat_t,  "sat_info",   "Saturacao.")

        # Diff tab
        dc = ttk.Frame(diff_t); dc.pack(fill=tk.X, pady=2)
        ttk.Label(dc, text="X (col):").pack(side=tk.LEFT)
        ttk.Spinbox(dc, textvariable=self.diff_col_var, from_=0, to=9999, width=6,
                    command=self._update_phase_diff).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(dc, text="Y (linha):").pack(side=tk.LEFT)
        ttk.Spinbox(dc, textvariable=self.diff_line_var, from_=0, to=9999, width=6,
                    command=self._update_phase_diff).pack(side=tk.LEFT, padx=2)
        self.diff_fig, self.diff_ax = plt.subplots(1, 1, figsize=(6, 3))
        _dark(self.diff_fig, [self.diff_ax])
        self.diff_canvas = FigureCanvasTkAgg(self.diff_fig, master=diff_t)
        self.diff_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.diff_info = ttk.Label(diff_t, text="Reprojecao 3D variando Z.", foreground="gray")
        self.diff_info.pack()

    def _update_roi_image(self):
        if not self.left_imgs: return
        imgs = self.left_imgs if self.cam_view_var.get() == "left" else self.right_imgs
        idx = min(self.img_idx_var.get(), len(imgs) - 1)
        self.roi_ax.clear()
        self.roi_ax.imshow(imgs[idx], cmap="gray", aspect="auto", origin="upper")
        self.roi_ax.set_title(f"{self.cam_view_var.get()} img {idx}", color="white")
        if self.roi:
            x0, y0, x1, y1 = self.roi
            self.roi_ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                             linewidth=2, edgecolor="#ff4444",
                                             facecolor="#ff444422"))
        _dark(self.roi_fig, [self.roi_ax]); self.roi_ax.axis("off")
        self.roi_canvas.draw_idle()

    def _roi_press(self, e):
        if e.inaxes != self.roi_ax or e.button != 1: return
        self._roi_dragging = True; self._press_xy = (e.xdata, e.ydata)
        if self._roi_patch:
            try: self._roi_patch.remove()
            except Exception: pass
            self._roi_patch = None

    def _roi_motion(self, e):
        if not self._roi_dragging or e.inaxes != self.roi_ax or e.xdata is None: return
        x0, y0 = self._press_xy; w = e.xdata - x0; h = e.ydata - y0
        if self._roi_patch:
            try: self._roi_patch.remove()
            except Exception: pass
        self._roi_patch = Rectangle((x0, y0), w, h, linewidth=2,
                                     edgecolor="#ff4444", facecolor="#ff444418", linestyle="--")
        self.roi_ax.add_patch(self._roi_patch); self.roi_canvas.draw_idle()

    def _roi_release(self, e):
        if not self._roi_dragging: return
        self._roi_dragging = False
        if e.xdata is None: return
        x0, y0 = self._press_xy
        xi0 = int(min(x0, e.xdata)); xi1 = int(max(x0, e.xdata))
        yi0 = int(min(y0, e.ydata)); yi1 = int(max(y0, e.ydata))
        if xi1 - xi0 < 4 or yi1 - yi0 < 4: return
        self.roi = (xi0, yi0, xi1, yi1)
        self.roi_x0_var.set(str(xi0)); self.roi_y0_var.set(str(yi0))
        self.roi_x1_var.set(str(xi1)); self.roi_y1_var.set(str(yi1))
        self._update_roi_image()

    def _clear_roi(self): self.roi = None; self._roi_patch = None; self._update_roi_image()

    def _roi_full(self):
        if not self.left_imgs: return
        H, W = self.left_imgs[0].shape[:2]; self.roi = (0, 0, W, H)
        self.roi_x0_var.set("0"); self.roi_y0_var.set("0")
        self.roi_x1_var.set(str(W)); self.roi_y1_var.set(str(H))
        self._update_roi_image()

    def _apply_manual_roi(self):
        try:
            self.roi = (self._si(self.roi_x0_var.get()), self._si(self.roi_y0_var.get()),
                        self._si(self.roi_x1_var.get(), 100), self._si(self.roi_y1_var.get(), 100))
            self._update_roi_image()
        except Exception:
            messagebox.showerror("Erro", "ROI devem ser inteiros.")

    def _run_roi_analysis(self):
        if not self.left_imgs:
            messagebox.showwarning("Aviso", "Carregue imagens primeiro."); return
        method = self.method_var.get(); sat_thr = self._si(self.sat_thresh_var.get(), 245)
        H, W = self.left_imgs[0].shape[:2]
        x0, y0, x1, y1 = self.roi if self.roi else (0, 0, W, H)
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            messagebox.showerror("Erro", "ROI invalida."); return

        L = np.stack(self.left_imgs,  0).astype(np.float32)
        R = np.stack(self.right_imgs, 0).astype(np.float32)
        Lr = L[:, y0:y1, x0:x1]; Rr = R[:, y0:y1, x0:x1]
        Ls = Lr.std(0); Rs = Rr.std(0)
        Lsat = (Lr.max(0) >= sat_thr).astype(np.float32)
        Rsat = (Rr.max(0) >= sat_thr).astype(np.float32)

        def _d2(fig, axes_arr, canvas, imgs, titles, cmap="inferno", vrange=None):
            axes = list(np.array(axes_arr).ravel())
            for ax in axes: ax.clear(); ax.set_facecolor("#2a2a3e")
            for ax, img, title in zip(axes, imgs, titles):
                kw = {"cmap": cmap, "aspect": "auto"}
                if vrange: kw["vmin"], kw["vmax"] = vrange
                im = ax.imshow(img, **kw); ax.set_title(title, color="white"); ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)
            _dark(fig, axes); fig.tight_layout(); canvas.draw_idle()

        _d2(self.std_fig, self.std_axes, self.std_canvas, [Ls, Rs],
            [f"STD Esq.(N={len(self.left_imgs)})", f"STD Dir.(N={len(self.right_imgs)})"])
        self.std_info.config(
            text=f"ROI [{x0},{y0}]-[{x1},{y1}]  STD max Esq={Ls.max():.1f} Dir={Rs.max():.1f}",
            foreground="#88aaff")

        pct_l = Lsat.mean() * 100; pct_r = Rsat.mean() * 100
        _d2(self.sat_fig, self.sat_axes, self.sat_canvas, [Lsat, Rsat],
            [f"Sat.Esq.({pct_l:.1f}%)", f"Sat.Dir.({pct_r:.1f}%)"], cmap="Reds", vrange=(0, 1))
        self.sat_info.config(
            text=f"Sat: Esq {pct_l:.1f}%  Dir {pct_r:.1f}%",
            foreground="#ff8844" if max(pct_l, pct_r) > 5 else "#88aaff")

        # ── Phase map (grayscale for discontinuity visibility) ─────────────────
        axes_p = list(np.array(self.phase_axes).ravel())
        for ax in axes_p: ax.clear(); ax.set_facecolor("#2a2a3e")
        if method == "fringe" and self.fringe_abs_left is not None:
            for ax, arr, t in zip(axes_p,
                                  [self.fringe_abs_left[y0:y1, x0:x1],
                                   self.fringe_abs_right[y0:y1, x0:x1]],
                                  ["Fase Abs. Esq.(rad)", "Fase Abs. Dir.(rad)"]):
                im = ax.imshow(arr, cmap="gray", aspect="auto")
                ax.set_title(t, color="white"); ax.axis("off")
                self.phase_fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)
            self.phase_info.config(
                text="Fase absoluta (gray=descontinuidades visiveis).", foreground="#88aaff")
        elif method == "fringe":
            axes_p[0].text(0.5, 0.5, "Fase nao calculada.", color="#ffaa44",
                           ha="center", va="center", transform=axes_p[0].transAxes)
            self.phase_info.config(text="Fase nao disponivel.", foreground="orange")
        else:
            for ax, arr, t in zip(axes_p, [Ls, Rs], ["STD Esq.", "STD Dir."]):
                im = ax.imshow(arr, cmap="viridis", aspect="auto")
                ax.set_title(t, color="white"); ax.axis("off")
                self.phase_fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)
            self.phase_info.config(text=f"{method}: STD.", foreground="#88aaff")
        _dark(self.phase_fig, axes_p); self.phase_fig.tight_layout(); self.phase_canvas.draw_idle()

        # ── Modulation ─────────────────────────────────────────────────────────
        axes_m = list(np.array(self.mod_axes).ravel())
        for ax in axes_m: ax.clear(); ax.set_facecolor("#2a2a3e")
        if method == "fringe" and self.fringe_mod_left is not None:
            for ax, arr, t in zip(axes_m,
                                  [self.fringe_mod_left[y0:y1, x0:x1],
                                   self.fringe_mod_right[y0:y1, x0:x1]],
                                  ["Modulacao Esq.", "Modulacao Dir."]):
                im = ax.imshow(arr, cmap="inferno", aspect="auto")
                ax.set_title(t, color="white"); ax.axis("off")
                self.mod_fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)
            self.mod_info.config(text="Mapa de modulacao.", foreground="#88aaff")
        else:
            axes_m[0].text(0.5, 0.5, "Modulacao apenas p/ fringe.", color="#ffaa44",
                           ha="center", va="center", transform=axes_m[0].transAxes)
        _dark(self.mod_fig, axes_m); self.mod_fig.tight_layout(); self.mod_canvas.draw_idle()

        self._update_phase_diff()
        self.status_var.set(
            f"Analise ROI concluida. Sat Esq:{pct_l:.1f}% Dir:{pct_r:.1f}%")

    def _update_phase_diff(self):
        self.diff_ax.clear(); self.diff_ax.set_facecolor("#2a2a3e")
        method = self.method_var.get()
        if method != "fringe" or self.fringe_abs_left is None:
            self.diff_ax.text(0.5, 0.5, "Disponivel apenas p/ fringe com fase calc.",
                              color="#ffaa44", ha="center", va="center",
                              transform=self.diff_ax.transAxes)
            _dark(self.diff_fig, [self.diff_ax]); self.diff_canvas.draw_idle(); return
        x0, y0, x1, y1 = self.roi if self.roi else (
            0, 0, self.left_imgs[0].shape[1], self.left_imgs[0].shape[0])
        x_rel = self.diff_col_var.get(); y_rel = self.diff_line_var.get()
        if y_rel < 0 or y_rel >= (y1 - y0) or x_rel < 0 or x_rel >= (x1 - x0):
            self.diff_ax.text(0.5, 0.5, "Ponto (X,Y) fora da ROI.", color="#ff4444",
                              ha="center", va="center", transform=self.diff_ax.transAxes)
            _dark(self.diff_fig, [self.diff_ax]); self.diff_canvas.draw_idle(); return
        u = x0 + x_rel; v = y0 + y_rel
        z0_v = self._sf(self.gz0.get(), -100); z1_v = self._sf(self.gz1.get(), 100)
        try:
            kl = self.zscan.camera_params["left"]["kk"].cpu().numpy()
            rl = self.zscan.camera_params["left"]["r"].cpu().numpy()
            tl = self.zscan.camera_params["left"]["t"].cpu().numpy()
            inv_k = np.linalg.inv(kl); inv_r = np.linalg.inv(rl)
            d_gcs = inv_r @ (inv_k @ np.array([u, v, 1.0]))
            c_gcs = inv_r @ (-tl).flatten()
            z_vals = np.linspace(z0_v, z1_v, 300)
            s_vals = (z_vals - c_gcs[2]) / (d_gcs[2] + 1e-12)
            pts_3d = c_gcs[None, :] + s_vals[:, None] * d_gcs[None, :]
            pts_t  = torch.from_numpy(pts_3d.astype(np.float32)).to(self.zscan.device)
            uv_l   = self.zscan.transform_gcs2ccs(pts_t, "left")
            uv_r   = self.zscan.transform_gcs2ccs(pts_t, "right")
            uv_l_np = uv_l.cpu().numpy(); uv_r_np = uv_r.cpu().numpy()
            import scipy.ndimage
            pl = scipy.ndimage.map_coordinates(
                self.fringe_abs_left,  [uv_l_np[:, 1], uv_l_np[:, 0]], order=1, mode="nearest")
            pr = scipy.ndimage.map_coordinates(
                self.fringe_abs_right, [uv_r_np[:, 1], uv_r_np[:, 0]], order=1, mode="nearest")
            diff = np.abs(pl - pr)
            self.diff_ax.plot(z_vals, diff, color="#ff4444", lw=1.5, label="|Esq−Dir|")
            mi = np.argmin(diff)
            self.diff_ax.axvline(z_vals[mi], color="#44ff88", lw=0.8, linestyle="--", alpha=0.7)
            self.diff_ax.plot(z_vals[mi], diff[mi], "wo", ms=5, label=f"Min Z={z_vals[mi]:.1f} mm")
            self.diff_ax.set_title(f"Fase via Raio 3D  (u={u}, v={v})", color="white")
            self.diff_ax.set_xlabel("Z (mm)", color="gray")
            self.diff_ax.set_ylabel("|Δφ| (rad)", color="gray")
            self.diff_ax.grid(True, alpha=0.3)
            self.diff_ax.legend(facecolor="#2a2a3e", edgecolor="gray", labelcolor="white")
        except Exception as e:
            self.diff_ax.text(0.5, 0.5, f"Erro: {e}", color="#ff4444",
                              ha="center", va="center", transform=self.diff_ax.transAxes)
        _dark(self.diff_fig, [self.diff_ax])
        self.diff_fig.tight_layout(); self.diff_canvas.draw_idle()

    # ── Tab: Projection ───────────────────────────────────────────────────────
    def _build_projection_tab(self, parent) -> None:
        ctrl = ttk.Frame(parent, width=330)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8)); ctrl.pack_propagate(False)
        plots = ttk.Frame(parent); plots.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r = [0]

        def _rng(lbl, v0, v1):
            ttk.Label(ctrl, text=lbl).grid(row=r[0], column=0, sticky="w")
            ttk.Entry(ctrl, textvariable=v0, width=8).grid(row=r[0], column=1, sticky="w", padx=2)
            ttk.Label(ctrl, text="a").grid(row=r[0], column=2)
            ttk.Entry(ctrl, textvariable=v1, width=8).grid(row=r[0], column=3, sticky="w", padx=2)
            r[0] += 1

        ttk.Label(ctrl, text="Grade 3D para Projecao",
                  style="Hdr.TLabel").grid(row=r[0], column=0, columnspan=4, sticky="w"); r[0] += 1
        _rng("X (mm):", self.gx0, self.gx1)
        _rng("Y (mm):", self.gy0, self.gy1)
        _rng("Z (mm):", self.gz0, self.gz1)
        ttk.Label(ctrl, text="Passo XY:").grid(row=r[0], column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.gxy_step, width=8).grid(
            row=r[0], column=1, sticky="w"); r[0] += 1
        ttk.Label(ctrl, text="Passo Z:").grid(row=r[0], column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.gz_step, width=8).grid(
            row=r[0], column=1, sticky="w"); r[0] += 1

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=r[0], column=0, columnspan=4, sticky="ew", pady=4); r[0] += 1
        for t, v in [("Grade uniforme", "grid"), ("Nuvem CSV", "csv")]:
            ttk.Radiobutton(ctrl, text=t, variable=self.proj_source_var, value=v).grid(
                row=r[0], column=0, columnspan=4, sticky="w"); r[0] += 1
        cf = ttk.Frame(ctrl); cf.grid(row=r[0], column=0, columnspan=4, sticky="ew"); r[0] += 1
        ttk.Entry(cf, textvariable=self.cloud_csv_var, width=26).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(cf, text="...", width=3,
                   command=lambda: self._browse_file(
                       self.cloud_csv_var, [("CSV", "*.csv")])).pack(side=tk.LEFT)

        ttk.Separator(ctrl, orient="horizontal").grid(
            row=r[0], column=0, columnspan=4, sticky="ew", pady=4); r[0] += 1
        ttk.Label(ctrl, text="Img fundo:").grid(row=r[0], column=0, columnspan=2, sticky="w")
        ttk.Spinbox(ctrl, textvariable=self.proj_img_idx, from_=0, to=999, width=5).grid(
            row=r[0], column=2, columnspan=2, sticky="w"); r[0] += 1
        ttk.Label(ctrl, text="Colorir por:").grid(row=r[0], column=0, sticky="w")
        ttk.Combobox(ctrl, textvariable=self.proj_color_var, width=12,
                     values=["Z", "X", "Y", "Correlacao"], state="readonly").grid(
                         row=r[0], column=1, columnspan=3, sticky="w"); r[0] += 1
        ttk.Separator(ctrl, orient="horizontal").grid(
            row=r[0], column=0, columnspan=4, sticky="ew", pady=4); r[0] += 1
        self.btn_proj = ttk.Button(ctrl, text="  Projetar  ", command=self._run_projection)
        self.btn_proj.grid(row=r[0], column=0, columnspan=4, sticky="ew", pady=2); r[0] += 1
        ttk.Button(ctrl, text="Exportar PNG", command=self._export_projection).grid(
            row=r[0], column=0, columnspan=4, sticky="ew")

        self.proj_fig, self.proj_axes = plt.subplots(1, 2, figsize=(10, 5))
        _dark(self.proj_fig, list(self.proj_axes))
        self.proj_canvas = FigureCanvasTkAgg(self.proj_fig, master=plots)
        self.proj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.proj_canvas, plots).update()

    def _run_projection(self):
        if self.zscan is None or not self.left_imgs:
            messagebox.showwarning("Aviso", "Carregue imagens primeiro."); return
        self.status_var.set("Calculando projecao..."); self.btn_proj.config(state=tk.DISABLED)

        def _w():
            try:
                src = self.proj_source_var.get(); ck = self.proj_color_var.get()
                if src == "csv":
                    cp = Path(self.cloud_csv_var.get())
                    if not cp.exists():
                        raise FileNotFoundError(f"CSV nao encontrado: {cp}")
                    data = np.loadtxt(cp, delimiter=",", skiprows=1)
                    if data.ndim == 1: data = data.reshape(1, -1)
                    xyz_np  = data[:, :3].astype(np.float32)
                    corr_np = (data[:, 3].astype(np.float32)
                               if data.shape[1] > 3 else np.ones(len(data), np.float32))
                else:
                    x = np.arange(self._sf(self.gx0.get(), -100), self._sf(self.gx1.get(), 500) + .01,
                                  max(self._sf(self.gxy_step.get(), 2.), .1))
                    y = np.arange(self._sf(self.gy0.get(), -100), self._sf(self.gy1.get(), 400) + .01,
                                  max(self._sf(self.gxy_step.get(), 2.), .1))
                    z = np.arange(self._sf(self.gz0.get(), -100), self._sf(self.gz1.get(), 100) + .01,
                                  max(self._sf(self.gz_step.get(), 2.), .1))
                    zm = z[len(z) // 2: len(z) // 2 + 1]
                    XX, YY, ZZ = np.meshgrid(x, y, zm, indexing="ij")
                    xyz_np  = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1).astype(np.float32)
                    corr_np = np.zeros(len(xyz_np), np.float32)

                pts = torch.from_numpy(xyz_np).to(self.zscan.device)
                uv_l, ml = self.zscan.transform_gcs2ccs(pts, "left",  image_shape=self.zscan.left_images.shape[1:])
                uv_r, mr = self.zscan.transform_gcs2ccs(pts, "right", image_shape=self.zscan.right_images.shape[1:])
                uv_l_np = uv_l.cpu().numpy(); uv_r_np = uv_r.cpu().numpy()
                ml_np   = ml.cpu().numpy();   mr_np   = mr.cpu().numpy()
                col_idx = {"X": 0, "Y": 1, "Z": 2}.get(ck, 2)
                cv = xyz_np[:, col_idx] if ck in ("X", "Y", "Z") else corr_np
                self.root.after(0, self._upd_proj, uv_l_np, uv_r_np, ml_np, mr_np, cv)
            except Exception as exc:
                self.root.after(0, self._upd_proj_err, str(exc))

        threading.Thread(target=_w, daemon=True).start()

    def _upd_proj(self, uv_l, uv_r, ml, mr, cv):
        self.btn_proj.config(state=tk.NORMAL)
        axl, axr = self.proj_axes
        axl.clear(); axr.clear()
        axl.set_facecolor("#2a2a3e"); axr.set_facecolor("#2a2a3e")
        idx = min(self.proj_img_idx.get(), len(self.left_imgs) - 1)
        if self.left_imgs:
            axl.imshow(self.left_imgs[idx],  cmap="gray", aspect="auto")
            axr.imshow(self.right_imgs[idx], cmap="gray", aspect="auto")
        vmin, vmax = cv.min(), cv.max()
        if vmax - vmin < 1e-8: vmax = vmin + 1.
        norm_cv = (cv - vmin) / (vmax - vmin)
        if ml.sum() > 0:
            axl.scatter(uv_l[ml, 0], uv_l[ml, 1],
                        c=plt.cm.plasma(norm_cv[ml]), s=3, alpha=0.75, linewidths=0)
        if mr.sum() > 0:
            axr.scatter(uv_r[mr, 0], uv_r[mr, 1],
                        c=plt.cm.plasma(norm_cv[mr]), s=3, alpha=0.75, linewidths=0)
        axl.set_title(f"Left  ({ml.sum()} pts)", color="white"); axl.axis("off")
        axr.set_title(f"Right ({mr.sum()} pts)", color="white"); axr.axis("off")
        _dark(self.proj_fig, [axl, axr]); self.proj_fig.tight_layout()
        self.proj_canvas.draw_idle()
        self.status_var.set(f"Projecao: L={ml.sum()} R={mr.sum()} pts visíveis.")

    def _upd_proj_err(self, msg):
        self.btn_proj.config(state=tk.NORMAL)
        self.status_var.set("Erro projecao.")
        messagebox.showerror("Erro", msg)

    def _export_projection(self):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.proj_fig.savefig(p, dpi=150, bbox_inches="tight")
            self.status_var.set(f"Exportado: {p}")

    # ── Tab: Filtros ──────────────────────────────────────────────────────────
    def _build_filter_tab(self, parent) -> None:
        top = ttk.Frame(parent); top.pack(fill=tk.X, pady=2)
        ctrl = ttk.LabelFrame(top, text=" Configuração "); ctrl.pack(side=tk.LEFT, padx=4)
        r = [0]

        def _e(lbl, var, w=8):
            ttk.Label(ctrl, text=lbl).grid(row=r[0], column=0, sticky="w")
            ttk.Entry(ctrl, textvariable=var, width=w).grid(
                row=r[0], column=1, sticky="w", padx=2); r[0] += 1

        _e("Threshold:",     self.filt_thresh_var)
        _e("Raio Eucl.:",    self.filt_radius_var)
        _e("Min. Vizinhos:", self.filt_minn_var)
        _e("Bounds STD/Mod:",self.filt_bounds_var)
        for k, t in [("filt_use_uv", "Filtro UV"), ("filt_use_std", "Filtro STD"),
                     ("filt_use_euclidean", "Filtro Euclidiano")]:
            ttk.Checkbutton(ctrl, text=t, variable=self.runner_vars[k],
                            command=self._on_filter_change).grid(
                                row=r[0], column=0, columnspan=2, sticky="w"); r[0] += 1

        load_frame = ttk.LabelFrame(top, text=" Nuvem CSV ")
        load_frame.pack(side=tk.LEFT, padx=8)
        cf = ttk.Frame(load_frame); cf.pack(pady=2)
        ttk.Entry(cf, textvariable=self.cloud_csv_var, width=30).pack(side=tk.LEFT)
        ttk.Button(cf, text="...", width=3,
                   command=lambda: self._browse_file(
                       self.cloud_csv_var, [("CSV", "*.csv")])).pack(side=tk.LEFT)
        ttk.Button(load_frame, text="Carregar Nuvem",
                   command=self._load_cloud).pack(pady=2)
        self.cloud_info_lbl = ttk.Label(load_frame, text="—", foreground="gray")
        self.cloud_info_lbl.pack(anchor="w", padx=4)

        info_frame = ttk.Frame(top); info_frame.pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Label(info_frame, text="Contagens:").pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.filt_counts_var,
                  foreground="#88aaff", wraplength=400).pack(anchor="w")

        # ── Projection sub-plot (filtered points on left image) ───────────────
        bot = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        bot.pack(fill=tk.BOTH, expand=True)
        pf = ttk.Frame(bot); bot.add(pf, weight=1)
        ff = ttk.Frame(bot); bot.add(ff, weight=2)

        ttk.Label(pf, text="Img fundo:").pack(anchor="w", padx=4)
        ttk.Spinbox(pf, textvariable=self.proj_img_idx, from_=0, to=999, width=5).pack(anchor="w", padx=4)
        self.filt_proj_fig, self.filt_proj_ax = plt.subplots(1, 1, figsize=(5, 4))
        _dark(self.filt_proj_fig, [self.filt_proj_ax])
        self.filt_proj_canvas = FigureCanvasTkAgg(self.filt_proj_fig, master=pf)
        self.filt_proj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.filt_proj_canvas, pf).update()

        self.filt_fig, faxes = plt.subplots(2, 2, figsize=(7, 5))
        _dark(self.filt_fig, list(faxes.ravel()))
        self.filt_hist_ax = faxes[0, 0]; self.filt_xy_ax = faxes[0, 1]
        self.filt_xz_ax   = faxes[1, 0]; self.filt_yz_ax = faxes[1, 1]
        self.filt_canvas  = FigureCanvasTkAgg(self.filt_fig, master=ff)
        self.filt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.filt_canvas, ff).update()

        # Bind filter change events
        for v in [self.filt_thresh_var, self.filt_radius_var,
                  self.filt_minn_var, self.filt_bounds_var]:
            v.trace_add("write", self._on_filter_change)

    def _load_cloud(self):
        cp = Path(self.cloud_csv_var.get())
        if not cp.exists():
            messagebox.showerror("Erro", f"Nao encontrado: {cp}"); return
        try:
            data = np.loadtxt(cp, delimiter=",", skiprows=1)
            if data.ndim == 1: data = data.reshape(1, -1)
            self.xyz_cloud  = data[:, :3].astype(np.float64)
            self.corr_cloud = (data[:, 3].astype(np.float64)
                               if data.shape[1] > 3 else np.ones(len(data)))
            self.cloud_info_lbl.config(text=f"{len(data)} pontos.", foreground="#22cc66")
            self.status_var.set(f"Nuvem: {len(data)} pontos.")
            self._on_filter_change()
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))

    def _on_filter_change(self, *_):
        if self.xyz_cloud is None: return
        if self._filter_debounce_id is not None:
            self.root.after_cancel(self._filter_debounce_id)
        self._filter_debounce_id = self.root.after(150, self._trigger_filter)

    def _trigger_filter(self):
        self._filter_debounce_id = None
        if self._filter_busy:
            self._filter_pending = True; return
        self._filter_busy = True
        threading.Thread(target=self._apply_filters, daemon=True).start()

    def _apply_filters(self):
        try:
            xyz  = self.xyz_cloud.copy(); corr = self.corr_cloud.copy()
            method = self.method_var.get(); n0 = len(xyz)
            thr = self.filt_thresh_var.get()
            if method == "fringe":
                mask = corr < thr
            else:
                cmax = corr.max() if corr.max() > 0 else 1.
                mask = corr > (thr * cmax)
            xyz = xyz[mask]; corr = corr[mask]; n1 = len(xyz)
            n2 = n1
            if self.filt_uv_var.get() and self.zscan is not None and HAS_TORCH and n1 > 0:
                try:
                    xt = torch.tensor(xyz, dtype=torch.float32, device=self.zscan.device)
                    _, ml = self.zscan.transform_gcs2ccs(xt, "left",  image_shape=self.zscan.left_images.shape[1:])
                    _, mr = self.zscan.transform_gcs2ccs(xt, "right", image_shape=self.zscan.right_images.shape[1:])
                    m = (ml & mr).cpu().numpy(); xyz = xyz[m]; corr = corr[m]; n2 = len(xyz)
                except Exception: pass
            n3 = n2
            if self.filt_std_var.get() and self.zscan is not None and HAS_TORCH and n2 > 0:
                try:
                    xt = torch.tensor(xyz,  dtype=torch.float32, device=self.zscan.device)
                    ct = torch.tensor(corr, dtype=torch.float32, device=self.zscan.device)
                    xt, ct, _ = self.zscan.std_mask_points(xt, ct,
                                                            bounds=self.filt_bounds_var.get(),
                                                            method=method)
                    xyz = xt.cpu().numpy(); corr = ct.cpu().numpy(); n3 = len(xyz)
                except Exception: pass
            n4 = n3
            if self.filt_eucl_var.get() and self.zscan is not None and HAS_TORCH and n3 > 10:
                try:
                    xt = torch.tensor(xyz,  dtype=torch.float32, device=self.zscan.device)
                    ct = torch.tensor(corr, dtype=torch.float32, device=self.zscan.device)
                    xt, ct, _ = self.zscan.euclidean_filter(
                        xt, ct,
                        min_neighbors=self.filt_minn_var.get(),
                        radius=self.filt_radius_var.get())
                    xyz = xt.cpu().numpy(); corr = ct.cpu().numpy(); n4 = len(xyz)
                except Exception: pass
            uv_l = uv_mask = None
            if self.zscan is not None and HAS_TORCH and n4 > 0:
                try:
                    xt = torch.tensor(xyz, dtype=torch.float32, device=self.zscan.device)
                    _uv, _m = self.zscan.transform_gcs2ccs(xt, "left",
                                                            image_shape=self.zscan.left_images.shape[1:])
                    uv_l = _uv.cpu().numpy(); uv_mask = _m.cpu().numpy()
                except Exception: pass
            pct = (n4 / n0 * 100) if n0 > 0 else 0.
            counts = (f"Total:{n0} | Thresh:{n1} | UV:{n2} | "
                      f"STD:{n3} | Eucl:{n4}  ({pct:.1f}%)")
            self.root.after(0, self._upd_filter_plots,
                            xyz, corr, uv_l, uv_mask, self.xyz_cloud, self.corr_cloud, counts)
        except Exception as exc:
            print(f"[Filter] {exc}")
        finally:
            self._filter_busy = False
            if self._filter_pending:
                self._filter_pending = False
                self.root.after(0, self._trigger_filter)

    def _upd_filter_plots(self, xyz, corr, uv_l, uv_mask, xyz0, corr0, counts):
        self.filt_counts_var.set(counts)
        self.filt_proj_ax.clear(); self.filt_proj_ax.set_facecolor("#2a2a3e")
        if self.left_imgs:
            idx = min(self.proj_img_idx.get(), len(self.left_imgs) - 1)
            self.filt_proj_ax.imshow(self.left_imgs[idx], cmap="gray", aspect="auto")
        if uv_l is not None and uv_mask is not None and uv_mask.sum() > 0:
            cv = corr[uv_mask] if len(corr) == len(uv_mask) else np.ones(uv_mask.sum())
            vmin, vmax = cv.min(), cv.max()
            if vmax - vmin < 1e-8: vmax = vmin + 1.
            self.filt_proj_ax.scatter(uv_l[uv_mask, 0], uv_l[uv_mask, 1],
                                      c=plt.cm.plasma((cv - vmin) / (vmax - vmin)),
                                      s=3, alpha=0.75, linewidths=0)
        nv = int(uv_mask.sum()) if uv_mask is not None else 0
        self.filt_proj_ax.set_title(f"Esq. — {nv} vis / {len(xyz)} total", color="white")
        _dark(self.filt_proj_fig, [self.filt_proj_ax]); self.filt_proj_ax.axis("off")
        self.filt_proj_fig.tight_layout(); self.filt_proj_canvas.draw_idle()

        for ax in [self.filt_hist_ax, self.filt_xy_ax, self.filt_xz_ax, self.filt_yz_ax]:
            ax.clear(); ax.set_facecolor("#2a2a3e")
        if len(corr0) > 0:
            self.filt_hist_ax.hist(corr0, bins=60, color="steelblue",
                                   alpha=0.45, label="Original", density=True)
        if len(corr) > 0:
            self.filt_hist_ax.hist(corr,  bins=60, color="#ff8844",
                                   alpha=0.75, label="Filtrado",  density=True)
        self.filt_hist_ax.set_title("Histograma Correlacao", color="white")
        self.filt_hist_ax.legend(fontsize=7, labelcolor="white",
                                 facecolor="#2a2a3e", edgecolor="#444466")
        for sp in self.filt_hist_ax.spines.values(): sp.set_edgecolor("#444466")
        self.filt_hist_ax.tick_params(colors="#cccccc")

        def _sc(ax, xi, yi, xl, yl, title):
            if len(xyz) > 0:
                c = corr if len(corr) == len(xyz) else np.ones(len(xyz))
                ax.scatter(xyz[:, xi], xyz[:, yi], c=c, cmap="viridis", s=2, alpha=0.6)
                ax.set_xlabel(xl, color="#cccccc"); ax.set_ylabel(yl, color="#cccccc")
            else:
                ax.text(0.5, 0.5, "Sem pontos", color="gray",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, color="white"); ax.tick_params(colors="#cccccc")
            for sp in ax.spines.values(): sp.set_edgecolor("#444466")

        _sc(self.filt_xy_ax, 0, 1, "X[mm]", "Y[mm]", "XY")
        _sc(self.filt_xz_ax, 0, 2, "X[mm]", "Z[mm]", "XZ")
        _sc(self.filt_yz_ax, 1, 2, "Y[mm]", "Z[mm]", "YZ")
        self.filt_fig.tight_layout(); self.filt_canvas.draw_idle()

    # ── Tab: Log ──────────────────────────────────────────────────────────────
    def _build_log_tab(self, parent) -> None:
        ctrl = ttk.Frame(parent); ctrl.pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="🗑 Limpar Log", command=self._clear_log).pack(side=tk.LEFT, padx=4)
        self.log_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 9),
            bg="#1e1e1e", fg="#e0e0e0", insertbackground="white")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        for tag, color in [("info",    "#80c0ff"), ("success", "#50e050"),
                            ("warning", "#ffb030"), ("error",   "#ff6060"),
                            ("header",  "#ffff70")]:
            kw = {"foreground": color}
            if tag == "header": kw["font"] = ("Consolas", 9, "bold")
            self.log_text.tag_config(tag, **kw)

    def _clear_log(self): self.log_text.delete("1.0", tk.END)

    def _log(self, msg, tag="info"):
        self.log_text.insert(tk.END, msg, tag); self.log_text.see(tk.END)

    # ══════════════════════════════════════════════════════════════════════════
    # Runner
    # ══════════════════════════════════════════════════════════════════════════
    def _save_cfg(self):
        cfg = {k: v.get() for k, v in self.runner_vars.items()}
        save_config(cfg)
        self._log("Config salva.\n", "success")

    def _build_runner_script(self) -> str:
        rv = self.runner_vars

        def _b(k):
            v = rv[k]
            return v.get() if isinstance(v, tk.BooleanVar) else str(v.get()) == "True"

        def _s(k): return str(rv[k].get())

        # ── Validation ────────────────────────────────────────────────────────
        methods = [m for m in ["laser", "fringe", "correl"] if _b(f"method_{m}")]
        if not methods:
            raise ValueError("Selecione ao menos um metodo.")

        dataset_root = _s("dataset_root").strip()
        if not dataset_root or not Path(dataset_root).is_dir():
            raise ValueError("Informe o Dataset Root (pasta com subpastas de timestamp).")

        timestamps = self._get_selected_timestamps()
        if not timestamps:
            raise ValueError("Selecione ao menos um timestamp.")

        # Validate: each (timestamp, method) must have left/ and right/
        valid_pairs = []
        for ts in timestamps:
            for m in methods:
                p = Path(dataset_root) / ts / m
                if (p / "left").is_dir() and (p / "right").is_dir():
                    valid_pairs.append((ts, m))
        if not valid_pairs:
            raise ValueError(
                f"Nenhum par (timestamp/método) válido encontrado em {dataset_root}.\n"
                "Verifique se as subpastas left/ e right/ existem.")

        yaml     = _s("yaml_file")
        bf       = float(_s("black_factor"))
        use_c2f  = _b("use_c2f")
        g1xy     = float(_s("g1_xy")); g1z  = float(_s("g1_z"))
        g2xy     = float(_s("g2_xy")); g2z  = float(_s("g2_z"))
        dz_c     = float(_s("dz_coarse"))
        fine_win = float(_s("fine_window"))
        surf_fol = _b("surface_following")
        dx0 = _s("diag_x_min"); dx1 = _s("diag_x_max")
        dy0 = _s("diag_y_min"); dy1 = _s("diag_y_max")
        dz0 = _s("diag_z_min"); dz1 = _s("diag_z_max")
        px0 = _s("proc_x_min"); px1 = _s("proc_x_max")
        py0 = _s("proc_y_min"); py1 = _s("proc_y_max")

        def _pnl(s): return [int(x.strip()) for x in s.split(",") if x.strip()]

        # Per-method params dict
        mp = {}
        for m, is_f in [("laser", False), ("correl", False), ("fringe", True)]:
            mp[m] = dict(
                n   = _pnl(_s(f"{m}_n_imgs")),
                k   = _pnl(_s(f"{m}_kernels")),
                thr = float(_s(f"{m}_thresh")),
                r   = float(_s(f"{m}_radius")),
                mn  = int(_s(f"{m}_min_neighbors")),
                b   = float(_s(f"{m}_mask_bounds")),
            )
            if is_f:
                mp[m]["px_f"]  = int(_s("fringe_px_f"))
                mp[m]["steps"] = int(_s("fringe_steps"))

        # ── process_block: 16-space indent matches the for-k loop body ──────
        # Structure: for ts (4) > for method (8) > for ni (12) > for k (16)
        _i = "                "   # 16 spaces
        if use_c2f:
            process_block = (
                f"{_i}# C2F: coarse-to-fine + parabolic sub-voxel\n"
                f"{_i}print(f\"  k={{k}} N={{ni}} [C2F] dz={dz_c:.2f} win=±{fine_win:.1f}\")\n"
                f"{_i}Zscan.points3d(x_lim=GL['x'], y_lim=GL['y'], z_lim=GL['z'],\n"
                f"{_i}               xy_step={g1xy}, z_step={g1z})\n"
                f"{_i}xyz, corr, _ = Zscan.process_coarse_to_fine(\n"
                f"{_i}    Kx=k, Ky=k, stride=1,\n"
                f"{_i}    dz_coarse={dz_c}, fine_window_mm={fine_win},\n"
                f"{_i}    Nz_block_coarse=20, method=method,\n"
                f"{_i}    surface_following={surf_fol})\n"
            )
        else:
            process_block = (
                f"{_i}# Pass 1: coarse brute-force\n"
                f"{_i}print(f\"  k={{k}} N={{ni}} [BF] pass-1\")\n"
                f"{_i}Zscan.points3d(x_lim=GL['x'], y_lim=GL['y'], z_lim=GL['z'],\n"
                f"{_i}               xy_step={g1xy}, z_step={g1z})\n"
                f"{_i}xyz, corr, _ = Zscan.process_segmented_z(Kx=k, Ky=k, stride=1, Nz_block_voxels=20, method=method)\n"
                f"{_i}if xyz.numel() == 0: print('  Sem pontos.'); continue\n"
                f"{_i}m_thr = corr < pp['thr'] if method == 'fringe' else corr > pp['thr']\n"
                f"{_i}xyz = xyz[m_thr]; corr = corr[m_thr]\n"
                f"{_i}if muv:  xyz, corr, _ = Zscan.mask_uv_points(xyz, corr)\n"
                f"{_i}if mstd: xyz, corr, _ = Zscan.std_mask_points(xyz, corr, bounds=pp['b'], method=method)\n"
                f"{_i}if eucl: xyz, corr, _ = Zscan.euclidean_filter(xyz, corr, min_neighbors=pp['mn'], radius=pp['r'])\n"
                f"{_i}if xyz.numel() == 0: print('  Sem pontos apos filtros.'); continue\n"
                f"{_i}xl = (xyz[:,0].min().item(), xyz[:,0].max().item())\n"
                f"{_i}yl = (xyz[:,1].min().item(), xyz[:,1].max().item())\n"
                f"{_i}zl = (xyz[:,2].min().item(), xyz[:,2].max().item())\n"
                f"{_i}# Pass 2: fine brute-force\n"
                f"{_i}print(f\"  Pass-2 dxy={g2xy} dz={g2z}\")\n"
                f"{_i}Zscan.points3d(x_lim=xl, y_lim=yl, z_lim=zl, xy_step={g2xy}, z_step={g2z})\n"
                f"{_i}xyz, corr, _ = Zscan.process_segmented_z(Kx=k, Ky=k, stride=1, Nz_block_voxels=5, method=method)\n"
            )

        mode_tag = "C2F" if use_c2f else "BruteForce"
        return f"""# Auto-generated by gui_zscan_unified.py  (v2 — dataset-aware)
import sys, os, time
import numpy as np, torch, cv2
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from include.fringe_projection.include.stereo_fringe_process import FringeProcess
from include.SpatioCorrelation import PyTorchStereoCorrel

def _csv(fn, xyz, corr=None):
    if isinstance(xyz, torch.Tensor):  xyz  = xyz.cpu().numpy()
    if corr is not None and isinstance(corr, torch.Tensor): corr = corr.cpu().numpy()
    if corr is not None:
        np.savetxt(fn, np.hstack((xyz, corr[:, None] if corr.ndim == 1 else corr)),
                   delimiter=',', header='x,y,z,corr', comments='')
    else:
        np.savetxt(fn, xyz, delimiter=',', header='x,y,z', comments='')
    print(f"CSV: {{fn}}")

def ri(folder, n=None):
    files = sorted(Path(folder).iterdir())
    if n: files = files[:n]
    imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    return [i for i in imgs if i is not None]

YAML         = {repr(yaml)}
DATASET_ROOT = Path({repr(dataset_root)})
TIMESTAMPS   = {repr(timestamps)}
METHODS      = {repr(methods)}
bf           = {bf}
plot         = {_b('plot')}
debug        = {_b('debug')}
save_file    = {_b('save_file')}
eucl         = {_b('euclidian_filter')}
mstd         = {_b('mask_std')}
muv          = {_b('mask_uv')}
MODE         = {repr(mode_tag)}
GL           = {{'x': ({px0}, {px1}), 'y': ({py0}, {py1}), 'z': ({dz0}, {dz1})}}
P            = {repr(mp)}

if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_properties(0).name}}")
else:
    print("CPU mode — sem GPU")

Zscan = PyTorchStereoCorrel(yaml_file=YAML)

t_total = time.time()
for ts in TIMESTAMPS:
    for method in METHODS:
        method_dir = DATASET_ROOT / ts / method
        lp = method_dir / 'left';  rp = method_dir / 'right'
        if not lp.is_dir() or not rp.is_dir():
            print(f"[SKIP] {{ts}}/{{method}} — sem left/right")
            continue
        pp = P[method]
        print('=' * 60)
        print(f"[{{MODE}}] {{ts}} / {{method}}")
        out = DATASET_ROOT / ts / f"results-{{method}}"
        out.mkdir(parents=True, exist_ok=True)

        if method == 'fringe':
            lr_raw = ri(lp); rr_raw = ri(rp)
            lr_raw = [cv2.convertScaleAbs(i, alpha=bf) for i in lr_raw]
            rr_raw = [cv2.convertScaleAbs(i, alpha=bf) for i in rr_raw]
            fp = FringeProcess(camera_resolution=lr_raw[0].shape[::-1],
                               px_f=pp['px_f'], steps=pp['steps'])
            fp.images_left  = np.moveaxis(np.array(lr_raw), 0, -1)
            fp.images_right = np.moveaxis(np.array(rr_raw), 0, -1)
            al, ar, mod_l, mod_r = fp.calculate_abs_phi_images(visualize=debug)
            lrp = [al.cpu().numpy(), mod_l.cpu().numpy()]
            rrp = [ar.cpu().numpy(), mod_r.cpu().numpy()]

        for ni in pp['n']:
            if method != 'fringe':
                lr_raw = ri(lp, ni); rr_raw = ri(rp, ni)
                lrp = [cv2.convertScaleAbs(i, alpha=bf) for i in lr_raw]
                rrp = [cv2.convertScaleAbs(i, alpha=bf) for i in rr_raw]

            for k in pp['k']:
                t0 = time.time()
                Zscan.convert_images(lrp, rrp, apply_clahe=(method != 'fringe'), undist=True)
{process_block}
                if xyz.numel() == 0: print("  Sem pontos apos processo."); continue
                m_thr = corr < pp['thr'] if method == 'fringe' else corr > pp['thr']
                xyz = xyz[m_thr]; corr = corr[m_thr]
                if muv:  xyz, corr, _ = Zscan.mask_uv_points(xyz, corr)
                if mstd: xyz, corr, _ = Zscan.std_mask_points(xyz, corr, bounds=pp['b'], method=method)
                if eucl: xyz, corr, _ = Zscan.euclidean_filter(
                              xyz, corr,
                              min_neighbors=pp['mn'] + (10 if not {use_c2f} else 0),
                              radius=pp['r'])
                print(f"  Pontos finais: {{xyz.shape[0]}}")
                if save_file and xyz.numel() > 0:
                    _csv(out / f'{{ts}}-{{method}}-{{ni}}img-{{k}}k.csv', xyz, corr)
                if plot and xyz.numel() > 0:
                    Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], corr,
                                         title=f'{{method}} {{ts}} N={{ni}} k={{k}}')
                print(f"  Concluido em {{time.time()-t0:.1f}} s")
                torch.cuda.empty_cache()

print(f"\\nTotal: {{time.time()-t_total:.1f}} s")
"""

    def _start_execution(self):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Aviso", "Processo em execucao."); return
        try:
            script = self._build_runner_script()
        except ValueError as e:
            messagebox.showerror("Config", str(e)); return
        runner = Path(__file__).parent / ".gui_unified_runner_temp.py"
        runner.write_text(script, encoding="utf-8")
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        mode = "C2F+Parabólico" if self.runner_vars["use_c2f"].get() else "Força Bruta"
        self.status_var.set(f"Executando ({mode})...")
        self.status_lbl.config(foreground="#cc8800")
        self._log(f"\n{'='*60}\nIniciando ({mode})...\n{'='*60}\n", "header")
        py = self.runner_vars["python_path"].get() or sys.executable
        self.worker_thread = threading.Thread(target=self._run_worker,
                                              args=(py, str(runner)), daemon=True)
        self.worker_thread.start()

    def _run_worker(self, py, sp):
        try:
            self.process = subprocess.Popen(
                [py, "-u", sp], cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in self.process.stdout:
                ll = line.lower(); tag = "info"
                if "erro" in ll or "traceback" in ll or "exception" in ll: tag = "error"
                elif "concluido" in ll or "salvo" in ll or "sucesso" in ll or "csv:" in ll: tag = "success"
                elif "aviso" in ll or "warning" in ll: tag = "warning"
                elif "===" in line or "[c2f]" in ll: tag = "header"
                self.root.after(0, self._log, line, tag)
            self.process.wait(); ok = self.process.returncode == 0
            self.root.after(0, self._on_done, ok,
                            "Concluido!" if ok else f"Erro (codigo {self.process.returncode})")
        except Exception as e:
            self.root.after(0, self._on_done, False, f"Falha: {e}")

    def _on_done(self, ok, msg):
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if ok:
            self.status_var.set("Concluido")
            self.status_lbl.config(foreground="#009933")
            self._log(f"\n[OK] {msg}\n", "success")
            if self.runner_vars["save_file"].get():
                self._log("[Dica] CSVs gerados. Carregue-os em Filtros → Carregar Nuvem.\n", "info")
        else:
            self.status_var.set("Erro / Interrompido")
            self.status_lbl.config(foreground="#cc0000")
            self._log(f"\n[FALHA] {msg}\n", "error")

    def _stop_execution(self):
        if self.process and self.process.poll() is None:
            self._log("\nInterrompendo...\n", "warning")
            self.process.terminate()
        self.status_var.set("Interrompido")
        self.status_lbl.config(foreground="#cc0000")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _si(v, d=0):
        try: return int(float(v))
        except Exception: return d

    @staticmethod
    def _sf(v, d=0.0):
        try: return float(v)
        except Exception: return d


def main() -> None:
    root = tk.Tk()
    UnifiedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
