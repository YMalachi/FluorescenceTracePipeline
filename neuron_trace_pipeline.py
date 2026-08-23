"""
Video -> neuron detection -> trace extraction -> spike model, with live GUI.

Part 2 of the project: takes our own lab calcium-imaging .tif stacks and feeds
extracted traces into the spike-detection models benchmarked in Part 3
(SimpleBaseline / OASIS / MLspike; those live in src/*.m and scripts/*.m in
this repo).

STATUS: all 4 stages (detection, trace extraction, MATLAB model integration,
live GUI) are implemented and wired up in main(). See KNOWN LIMITATIONS below
before treating Stage 1's detection output as ground truth.

NOTE on the dev video (D:\062926\5- 20fps\...): this culture is >14 DIV,
older than the DIV14 originally planned for. Older cultures like this one
commonly already show cell colonies/clusters rather than only isolated single
somata, which is a plausible explanation for some of the elongated/irregular
ROIs in Stage 1's detection output (they may be real multi-cell structures,
not pure noise).

KNOWN LIMITATION -- detection accuracy is not fully verified: Detection here
has NOT been calibrated against a manual FIJI/ROI-Manager reference run,
since no screenshots or exported ROI list from an actual run on this video
were available. Visually reviewing the raw video (not the ΔF/F view -- see
launch_gui()) against the detected boxes, some of the 20 ROIs land on
clearly recognizable soma-like structures (e.g. #8, #9, #18 and the #11-13/
17 cluster) while several small ones near the frame edges (#1,#2,#3,#5,#6,#7)
don't land on any obvious structure and may be residual noise rather than
real cells. Two candidate automatic filters were tried and explicitly did
NOT solve this cleanly enough to trust blindly:
  - shape (eccentricity/solidity): didn't separate the groups -- some
    visually-plausible ROIs are highly elongated (e.g. #9: eccentricity
    0.95) while some questionable ones are fairly round (e.g. #10: 0.43).
  - local raw-intensity contrast (ROI mean F0 vs. its immediate surround):
    also didn't separate them -- several visually-plausible ROIs are
    actually DARKER than their local surround (e.g. #11: contrast -59),
    while several questionable ones are brighter than theirs.
This suggests the visual "does this look like a real cell" judgment is a
holistic spatial pattern, not reducible to a simple per-ROI geometric or
intensity threshold -- a more principled fix would need a structurally-aware
detector (e.g. blob detection on the raw F0 image, cross-checked against
ΔF/F activity) rather than a quick parameter tweak. Given the Aug 16 draft
deadline, this was deliberately left as a disclosed limitation rather than
an open-ended detector rewrite -- report this honestly rather than
presenting all 20 ROIs as equally reliable.

Reproduces, in Python, the manual FIJI pipeline used as reference:
  1. Load stack.
  2. Z-project (Average) -> F0.
  3. Stack / F0 (32-bit) -> F/F0.
  4. Max projection of F/F0 -> shows which cells lit up.
  5. Threshold (the FIJI notes omitted this step but Analyze Particles
     requires a binary image first) + Analyze-Particles-equivalent
     (connected components filtered by min area) -> ROI list.
  6. ROI Manager / Multi Measure equivalent -> extract_traces() (Stage 2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import subprocess
import sys

import numpy as np
import scipy.io
import tifffile
from scipy import ndimage
from skimage import filters, measure, morphology

# =============================================================================
# CONFIG -- edit this block to rerun on a new video. No need to touch code
# below this section for a normal run.
# =============================================================================

CONFIG = {
    # =========================================================================
    # ============ MODEL CHOICE -- pick the spike-inference model ===========
    # =========================================================================
    # "oasis"                         -> deconvolution-based (src/run_oasis_model.m).
    #                                    Scientifically the right default for THIS
    #                                    video: per our own Part 3 benchmarking
    #                                    (results/tables/model_results_summary.csv),
    #                                    OASIS leads at 20Hz+; this video is 20Hz.
    # "custom_derivative_amplitude"   -> our own rule-based model (src/run_simple_baseline.m,
    #                                    a.k.a. "SimpleBaseline" in the README/benchmark
    #                                    tables -- same model, this is just the
    #                                    descriptive name used here):
    #                                        event_score = derivative_weight * positive_derivative
    #                                                    + amplitude_weight  * smoothed_calcium
    #                                    Tune the weights in MODEL_PARAMS below.
    # "mlspike"                       -> inference-based (src/run_mlspike_model.m; toolbox
    #                                    cloned into external/spikes + external/brick).
    #
    # Per our own Part 3 benchmarking (results/tables/model_results_summary.csv):
    # OASIS leads at 20Hz+ (this video's rate); MLspike leads at 10Hz. The
    # custom model is a lighter-weight baseline. All three are wired up
    # identically end-to-end (detection -> traces -> MATLAB model -> GUI) --
    # switching which one runs on this video is just changing this one line.
    "MODEL_NAME": "custom_derivative_amplitude",             # "oasis" | "custom_derivative_amplitude" | "mlspike"


    # Per-model parameters, passed straight through to the corresponding
    # src/run_*_model.m function. Defaults taken from the existing MATLAB
    # scripts (scripts/OASIS_ONE_NEURON.m, README's SimpleBaseline example).
    "MODEL_PARAMS": {
        "oasis": {
            "ar_model": "ar1",
            "method": "foopsi",
            "optimize_b": True,
            "optimize_pars": True,
            # oasis_threshold_z=4.5 (raised from the README/script default of
            # 1.5) -- swept 1.5..5.0 on ROI #1 and picked the value that
            # isolates exactly the two visually obvious transients (t=9.36s,
            # t=21.56s) instead of ~35 spurious events on flat baseline.
            # PROVISIONAL: eyeballed against one trace, not fit against
            # ground truth (none exists for this video). Revisit once/if a
            # real reference becomes available.
            "oasis_threshold_z": 4.5,
            "min_event_distance_sec": 0.10,
        },
        "custom_derivative_amplitude": {
            "smoothing_window_sec": 0.2,
            "derivative_weight": 0.7,
            "amplitude_weight": 0.3,
            "event_score_threshold": 0.9,
            "min_event_distance_sec": 0.10,
        },
        "mlspike": {
            # Defaults taken from scripts/MLSPIKE_ONE_NEURON.m, where they're
            # explicitly noted as "initial values only, tune later if the
            # model works" -- inherited here with the same caveat. Untested
            # against our own ΔF/F traces (very different scale/shape from
            # the mock benchmark data these were picked against).
            "use_autocalibration": False,
            "a": 0.07,
            "tau": 1.0,
            "sigma": 0.02,
            "saturation": 0.1,
            "drift_parameter": 0.01,
            # Only used if use_autocalibration=True:
            "amin": 0.02,
            "amax": 0.20,
            "taumin": 0.20,
            "taumax": 2.00,
        },
    },
    # =========================================================================

    # --- Input ---

    "VIDEO_PATH": r"D:\062926\6-20fps\6-20fps_MMStack_Default.ome.tif",
    "SAMPLING_RATE_HZ": 20.0,          # expected acquisition rate for this video
    "FPS_MISMATCH_TOLERANCE": 0.05,    # relative tolerance before flagging a mismatch

    # --- Neuron detection ---
    "THRESHOLD_METHOD": "otsu",        # "otsu" | "default" (ImageJ-style IsoData)
    "MIN_ROI_SIZE_PX": 30,             # Analyze Particles min-size filter equivalent
    "MAX_ROI_SIZE_PX": 1000,           # Analyze Particles max-size filter equivalent
                                        # (rejects large glare/out-of-focus blobs, not real somata)
    "SPATIAL_SMOOTH_SIGMA_PX": 2.0,    # per-frame Gaussian blur applied BEFORE the max
                                        # projection, to suppress independent per-pixel shot
                                        # noise (500 frames means raw per-pixel max is an
                                        # extreme-value statistic and picks up noise almost
                                        # everywhere without this). 0 disables.

    # --- Neuron selection for downstream display / model input (Stage 2+) ---
    "NEURON_SELECTION_MODE": "all",    # "all" | "top_1" | "top_k"
    "TOP_K": 5,

    "MATLAB_EXECUTABLE": "matlab",     # must be on PATH; verified working (R2025b)

    # --- Live GUI (Stage 4b) ---
    # "interactive" opens a real matplotlib window (plt.show()) -- use this
    # when running locally on a machine with a display attached (Guy's
    # desktop). "save_video" renders headlessly to an .mp4 instead -- used
    # here for review in an environment with no attached display; switch to
    # "interactive" for the actual live demo.
    "GUI_PLAYBACK_MODE": "interactive",   # "interactive" | "save_video"
    "GUI_VIDEO_FILENAME": "stage4b_live_gui_demo.mp4",   # written inside this run's output folder
    "GUI_PLAYBACK_SPEED": 2.0,         # 1.0 = real-time (matches SAMPLING_RATE_HZ); >1 = faster
    # Exported via OpenCV's mp4v encoder (opencv-python is already a
    # dependency; no ffmpeg install needed -- verified it produces a valid,
    # re-readable .mp4). mp4/H.264-family compression handles this grainy
    # grayscale content far better than GIF/LZW did (an earlier GIF version
    # of this was 353MB at full res/all frames; a same-quality .mp4 does not
    # need anywhere near that much subsampling). None = every frame, full dpi.
    "GUI_VIDEO_MAX_FRAMES": None,
    "GUI_VIDEO_DPI": 80,               # 100 looks marginally crisper but ~44MB vs ~21MB at 80
    # interactive mode also saves to disk: a snapshot PNG of whatever frame
    # was showing when you close the window, AND (after closing) the same
    # .mp4 that save_video mode produces -- so an interactive run always
    # leaves a shareable artifact behind, not just a live window.
    "GUI_SNAPSHOT_FILENAME": "stage4b_interactive_snapshot.png",  # written inside this run's output folder

    # --- Output ---
    # Base folder for this video. Every run (python neuron_trace_pipeline.py)
    # creates its own timestamped subfolder underneath -- OUTPUT_DIR/
    # YYYYMMDD_HHMMSS/ -- holding everything that run produced (overlay,
    # traces plot, matlab_io/, events/, the GUI mockup/video/snapshot,
    # pipeline_cache.pkl), so re-running on the same video never overwrites
    # a previous run. `--gui-only` locates and reuses the MOST RECENT such
    # subfolder (by timestamp) rather than needing a path spelled out.
    "OUTPUT_DIR": "results/python_pipeline_6-20fps",
}

PIPELINE_CACHE_FILENAME = "pipeline_cache.pkl"

# Maps CONFIG["MODEL_NAME"] (the friendly name used in this script and in
# MODEL_PARAMS) to the model_name string the MATLAB bridge scripts
# (scripts/run_model_*_from_files.m) actually dispatch on internally, which
# matches the underlying src/run_*_model.m function names. Only
# "custom_derivative_amplitude" differs -- it's our display name for what's
# implemented as run_simple_baseline.m / "SimpleBaseline" in the README and
# benchmark tables (results/tables/model_results_summary.csv), so results
# stay traceable back to the benchmarking work under its original name.
MODEL_DISPATCH = {
    "oasis": "oasis",
    "custom_derivative_amplitude": "simple_baseline",
    "mlspike": "mlspike",
}

REPO_ROOT = Path(__file__).resolve().parent


# =============================================================================
# Stage 1: neuron detection
# =============================================================================

@dataclass
class Roi:
    id: int
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid: tuple[float, float]    # (row, col)
    area_px: int
    coords: np.ndarray               # (N, 2) array of (row, col) pixel indices in the ROI


def load_stack(video_path: str) -> tuple[np.ndarray, tifffile.TiffFile]:
    """Load a .tif stack as (T, Y, X). Keeps the TiffFile handle open for metadata."""
    tf = tifffile.TiffFile(video_path)
    stack = tf.asarray()
    if stack.ndim != 3:
        raise ValueError(f"Expected a (T, Y, X) stack, got shape {stack.shape}")
    return stack.astype(np.float32), tf


def check_frame_rate(tf: tifffile.TiffFile, configured_hz: float, tolerance: float) -> float | None:
    """
    Sanity-check the configured sampling rate against the rate derivable from
    the file. MicroManager's own "Interval_ms" field in the OME/ImageJ headers
    is frequently a placeholder (e.g. fixed at 1.0 ms) and not the true
    acquisition rate, so we prefer the per-frame 'ElapsedTime-ms' timestamps
    MicroManager stamps on each page when available, and fall back to the
    header interval otherwise. Returns the measured Hz, or None if nothing
    could be derived (in which case the configured rate is used as-is).
    """
    measured_hz = None

    try:
        elapsed_ms = []
        for page in tf.pages:
            tag = page.tags.get("MicroManagerMetadata")
            if tag is None:
                break
            elapsed_ms.append(tag.value["ElapsedTime-ms"])
        if len(elapsed_ms) >= 2:
            total_ms = elapsed_ms[-1] - elapsed_ms[0]
            n_intervals = len(elapsed_ms) - 1
            measured_hz = 1000.0 * n_intervals / total_ms
    except Exception:
        measured_hz = None

    if measured_hz is None:
        try:
            ome = tf.ome_metadata
            if ome and "TimeIncrement=" in ome:
                import re
                m = re.search(r'TimeIncrement="([\d.]+)"', ome)
                if m:
                    increment_ms = float(m.group(1))
                    measured_hz = 1000.0 / increment_ms
                    warnings.warn(
                        "Using OME TimeIncrement header for frame rate; this field "
                        "is often an uncalibrated MicroManager placeholder, not a "
                        "measured value. Prefer per-frame timestamps when available."
                    )
        except Exception:
            measured_hz = None

    if measured_hz is None:
        print("[frame rate] Could not derive an actual frame rate from the file; "
              f"proceeding with configured SAMPLING_RATE_HZ={configured_hz}.")
        return None

    rel_diff = abs(measured_hz - configured_hz) / configured_hz
    if rel_diff > tolerance:
        print(
            f"[frame rate] *** MISMATCH *** configured SAMPLING_RATE_HZ={configured_hz} Hz "
            f"but measured rate from file timestamps is {measured_hz:.3f} Hz "
            f"({rel_diff * 100:.1f}% off, tolerance {tolerance * 100:.0f}%). "
            "Sampling rate affects which model/params are appropriate -- update "
            "CONFIG['SAMPLING_RATE_HZ'] or double-check the acquisition settings."
        )
    else:
        print(f"[frame rate] OK: configured {configured_hz} Hz vs measured {measured_hz:.3f} Hz "
              f"({rel_diff * 100:.1f}% diff, within {tolerance * 100:.0f}% tolerance).")

    return measured_hz


def compute_f0(stack: np.ndarray) -> np.ndarray:
    """Average projection across time (F0), per the reference FIJI pipeline."""
    return stack.mean(axis=0)


def compute_dff_stack(stack: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """ΔF/F(t) = stack / F0 - 1, computed per-pixel across the whole stack."""
    eps = 1e-6
    return stack / (f0[None, :, :] + eps) - 1.0


def threshold_image(image: np.ndarray, method: str) -> float:
    if method == "otsu":
        return filters.threshold_otsu(image)
    if method == "default":
        # Approximates ImageJ's "Default" (modified IsoData) method.
        return filters.threshold_isodata(image)
    raise ValueError(f"Unknown THRESHOLD_METHOD: {method!r}")


def detect_neurons(
    dff_stack: np.ndarray,
    threshold_method: str,
    min_roi_size_px: int,
    max_roi_size_px: int,
    spatial_smooth_sigma_px: float,
) -> tuple[list[Roi], np.ndarray, np.ndarray]:
    """
    Equivalent of: max projection -> threshold -> Analyze Particles (connected
    components, filtered by min/max area). Returns (rois, binary_mask, max_proj).

    The per-frame spatial blur (before the temporal max) matters: taking a raw
    per-pixel max over hundreds of frames is an extreme-value statistic, so
    without denoising each frame first, independent shot/read noise alone
    produces a "detection" at almost every pixel (verified empirically: >1900
    spurious ROIs on this video with no spatial smoothing, vs. ~20 plausible
    somata with sigma=2px).
    """
    smoothed_stack = dff_stack
    if spatial_smooth_sigma_px > 0:
        smoothed_stack = ndimage.gaussian_filter(
            dff_stack, sigma=(0, spatial_smooth_sigma_px, spatial_smooth_sigma_px)
        )
    max_proj = smoothed_stack.max(axis=0)

    thresh = threshold_image(max_proj, threshold_method)
    binary_mask = max_proj > thresh
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_roi_size_px)

    labels = measure.label(binary_mask)
    props = measure.regionprops(labels)

    # Stable IDs: sort in raster order (top-to-bottom, then left-to-right) so
    # the same video always yields the same numbering.
    props = [p for p in props if min_roi_size_px <= p.area <= max_roi_size_px]
    props.sort(key=lambda p: (p.centroid[0], p.centroid[1]))

    rois = [
        Roi(id=i + 1, bbox=p.bbox, centroid=p.centroid, area_px=int(p.area), coords=p.coords)
        for i, p in enumerate(props)
    ]
    return rois, binary_mask, max_proj


# =============================================================================
# Detection review plot (used for this checkpoint only)
# =============================================================================

def draw_detection_overlay(display_image: np.ndarray, rois: list[Roi], out_path: Path) -> None:
    # Backend is chosen once, centrally, in _configure_matplotlib_backend()
    # (called at the top of main()) -- NOT forced here. savefig() works the
    # same on every backend, so forcing Agg in this function would silently
    # lock the whole process out of showing an interactive window later
    # (this was a real bug: it broke Stage 4b's plt.show()).
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    p_lo, p_hi = np.percentile(display_image, (1, 99.5))
    vmin, vmax = float(p_lo), float(p_hi)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display_image, cmap="gray", vmin=vmin, vmax=vmax)

    color_coded = len(rois) <= 12
    cmap = plt.get_cmap("tab20" if len(rois) > 10 else "tab10")

    for i, roi in enumerate(rois):
        min_row, min_col, max_row, max_col = roi.bbox
        color = cmap(i % cmap.N) if color_coded else "yellow"
        rect = Rectangle(
            (min_col, min_row), max_col - min_col, max_row - min_row,
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            min_col, min_row - 3, f"#{roi.id}",
            color=color, fontsize=8, fontweight="bold",
            ha="left", va="bottom",
        )

    ax.set_title(f"Detected neurons: {len(rois)}  (threshold={CONFIG['THRESHOLD_METHOD']}, "
                 f"size=[{CONFIG['MIN_ROI_SIZE_PX']}, {CONFIG['MAX_ROI_SIZE_PX']}]px)")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Stage 2: trace extraction
# =============================================================================

def extract_traces(dff_stack: np.ndarray, rois: list[Roi]) -> np.ndarray:
    """
    Per-ROI ΔF/F(t): mean of dff_stack over each ROI's actual segmented
    pixels (not just its bounding box) at every frame.
    Uses the original (unsmoothed) dff_stack -- the spatial blur in
    detect_neurons was only an aid for locating ROIs robustly, not something
    that should leak into the extracted trace values.
    Returns an (n_rois, T) array.
    """
    n_rois = len(rois)
    n_frames = dff_stack.shape[0]
    traces = np.empty((n_rois, n_frames), dtype=np.float32)
    for i, roi in enumerate(rois):
        rows, cols = roi.coords[:, 0], roi.coords[:, 1]
        traces[i] = dff_stack[:, rows, cols].mean(axis=1)
    return traces


def select_neurons(rois: list[Roi], traces: np.ndarray, mode: str, top_k: int) -> list[int]:
    """
    Returns indices (into rois/traces) to keep, per NEURON_SELECTION_MODE.
    "Strongest" = highest peak |ΔF/F| over the trace.
    """
    if mode == "all":
        return list(range(len(rois)))
    peak_strength = np.abs(traces).max(axis=1)
    order = np.argsort(-peak_strength)
    if mode == "top_1":
        return [int(order[0])]
    if mode == "top_k":
        return sorted(int(i) for i in order[:top_k])
    raise ValueError(f"Unknown NEURON_SELECTION_MODE: {mode!r}")


def _normalize_traces_for_stacking(traces: np.ndarray, fill_fraction: float = 0.85) -> np.ndarray:
    """
    Scales each trace INDEPENDENTLY into its own [-fill_fraction/2, fill_fraction/2]
    band, using that trace's own true min/max (not a shared/global scale
    factor). This guarantees no trace's peak can ever cross into a
    neighboring row's space, regardless of how much amplitude varies across
    neurons -- a single global offset step (the previous approach) can't
    guarantee that once traces have noticeably different amplitudes.
    """
    lo = traces.min(axis=1, keepdims=True)
    hi = traces.max(axis=1, keepdims=True)
    spread = np.maximum(hi - lo, 1e-12)
    normalized = (traces - lo) / spread - 0.5  # exactly [-0.5, 0.5] per row
    return normalized * fill_fraction


def _draw_stacked_traces(
    ax,
    traces: np.ndarray,
    rois: list[Roi],
    sampling_rate_hz: float,
    color_coded: bool,
    cmap,
    neutral_color: str,
    predicted_events: np.ndarray | None = None,
    row_height: float = 1.0,
    fill_fraction: float = 0.85,
) -> None:
    """
    Ruled-notebook-style stacked traces: neuron #1 at the bottom, each on its
    own fixed-height row (never overlapping a neighbor -- see
    _normalize_traces_for_stacking), with a light baseline ruled under each
    row like lines on ruled paper. predicted_events (n_rois, T bool), if
    given, adds a small red marker (▼) per event, in the gap between rows so
    it can't be mistaken for crossing into either trace.
    """
    n_rois, n_frames = traces.shape
    t = np.arange(n_frames) / sampling_rate_hz
    normalized = _normalize_traces_for_stacking(traces, fill_fraction)

    for i, roi in enumerate(rois):
        row_offset = i * row_height
        color = cmap(i % cmap.N) if color_coded else neutral_color
        ax.axhline(row_offset, color="0.85", linewidth=0.6, zorder=0)  # ruled baseline
        ax.plot(t, normalized[i] + row_offset, color=color, linewidth=0.8, zorder=2)
        if predicted_events is not None:
            event_times = t[np.asarray(predicted_events[i]).astype(bool)]
            if len(event_times) > 0:
                ax.plot(
                    event_times,
                    np.full_like(event_times, row_offset + fill_fraction / 2 + 0.05 * row_height),
                    marker="v", linestyle="none", color="tab:red", markersize=4, zorder=3,
                )
        ax.text(t[-1] * 1.01, row_offset, f"#{roi.id}", color=color, fontsize=8, va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-row_height * 0.5, (n_rois - 1) * row_height + row_height * 0.5)
    ax.set_yticks([])


def plot_example_traces(
    traces: np.ndarray, rois: list[Roi], sampling_rate_hz: float, out_path: Path
) -> None:
    """
    Diagnostic-only stacked plot (bottom = neuron #1, per the eventual GUI's
    ordering convention) so extracted ΔF/F(t) traces can be sanity-checked
    before wiring up the model/GUI stages. Not the live GUI itself.
    """
    # See draw_detection_overlay() -- backend is set centrally, not here.
    import matplotlib.pyplot as plt

    n_rois, n_frames = traces.shape
    color_coded = n_rois <= 12
    cmap = plt.get_cmap("tab20" if n_rois > 10 else "tab10")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * n_rois)))

    _draw_stacked_traces(ax, traces, rois, sampling_rate_hz, color_coded, cmap, "steelblue")

    ax.set_xlabel("time (s)")
    ax.set_title(f"ΔF/F(t) per detected neuron (n={n_rois}, stacked #1 at bottom)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Stage 3: model integration, via a file-based subprocess bridge to the
# existing MATLAB models (src/run_oasis_model.m / run_simple_baseline.m /
# run_mlspike_model.m). We deliberately do NOT reimplement any model logic
# in Python -- MATLAB is invoked headlessly (`matlab -batch`) through
# scripts/run_model_from_files.m, exchanging a .mat file for the trace/params
# in and a .mat file for the result struct out.
# =============================================================================

def _to_matlab_path(path: Path) -> str:
    """MATLAB accepts forward slashes on Windows too; avoids backslash-escaping."""
    return str(path.resolve()).replace("\\", "/")


def _run_matlab(matlab_executable: str, matlab_call: str, log_prefix: str) -> subprocess.CompletedProcess:
    """
    Shared subprocess plumbing for both the single-ROI and batch MATLAB
    bridges: builds the `matlab -batch` command, runs it from scripts/,
    prints captured stdout/stderr (prefixed so the two bridges' output stays
    distinguishable in the log), and turns "matlab isn't on PATH" into a
    clear, actionable error instead of a raw FileNotFoundError traceback.
    Does NOT check the return code -- callers know their own success
    condition (e.g. also requires the expected output file to exist).
    """
    cmd = [matlab_executable, "-batch", matlab_call]
    scripts_dir = REPO_ROOT / "scripts"

    print(f"[{log_prefix}] cwd={scripts_dir}")
    print(f"[{log_prefix}] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(scripts_dir), capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not launch MATLAB executable {matlab_executable!r} -- is it on PATH? "
            f"Check CONFIG['MATLAB_EXECUTABLE']. Original error: {e}"
        ) from e

    print(f"[{log_prefix}] --- stdout ---")
    print(proc.stdout)
    if proc.stderr.strip():
        print(f"[{log_prefix}] --- stderr ---")
        print(proc.stderr)
    return proc


def run_model_via_matlab(
    calcium: np.ndarray,
    fps: float,
    model_name: str,
    params: dict,
    io_dir: Path,
    matlab_executable: str = "matlab",
) -> dict:
    """
    Runs one of the existing MATLAB models on a single trace via subprocess,
    through scripts/run_model_from_files.m. Raises RuntimeError (with the
    captured MATLAB stderr) if MATLAB exits non-zero -- e.g. because a
    required external toolbox is missing -- rather than swallowing the
    failure.

    Not used in main()'s default flow (run_model_batch_via_matlab covers all
    selected ROIs in one MATLAB session, which is both faster -- one MATLAB
    startup, not one per ROI -- and gives complete results); kept as a
    lower-level utility for ad-hoc single-trace debugging.
    """
    io_dir.mkdir(parents=True, exist_ok=True)
    input_path = io_dir / f"{model_name}_input.mat"
    output_path = io_dir / f"{model_name}_output.mat"

    if output_path.exists():
        output_path.unlink()  # so a stale file can't be mistaken for a fresh result

    scipy.io.savemat(
        str(input_path),
        {"calcium": np.asarray(calcium, dtype=np.float64), "fps": float(fps), "params": params},
    )

    matlab_call = (
        f"run_model_from_files('{_to_matlab_path(input_path)}', "
        f"'{_to_matlab_path(output_path)}', '{model_name}')"
    )
    proc = _run_matlab(matlab_executable, matlab_call, log_prefix="matlab")

    if proc.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"MATLAB model run failed (model={model_name!r}, returncode={proc.returncode}). "
            f"See stdout/stderr above. Input file left at {input_path} for inspection."
        )

    mat = scipy.io.loadmat(str(output_path))
    result = {
        k: np.squeeze(v) for k, v in mat.items() if not k.startswith("__")
    }
    return result


def run_model_batch_via_matlab(
    traces: np.ndarray,
    roi_ids: list[int],
    fps: float,
    model_name: str,
    params: dict,
    io_dir: Path,
    matlab_executable: str = "matlab",
) -> dict:
    """
    Runs the model on all ROIs in a single MATLAB session, via
    scripts/run_model_batch_from_files.m. Returns
    {"predicted_events": (n_rois, T) bool array, "n_predicted_events": (n_rois,) int array,
     "failed_roi_ids": list[int], "failed_roi_reasons": list[str]}.

    A per-ROI model failure (e.g. OASIS legitimately finding zero variance
    in its own deconvolved activity signal for a degenerate/borderline
    trace -- observed in practice once ROI count grew past ~30) does NOT
    abort the whole run; that ROI gets 0 events and shows up in
    failed_roi_ids/failed_roi_reasons instead of being silently dropped or
    crashing the other ROIs' results.
    """
    io_dir.mkdir(parents=True, exist_ok=True)
    input_path = io_dir / f"{model_name}_batch_input.mat"
    output_path = io_dir / f"{model_name}_batch_output.mat"

    if output_path.exists():
        output_path.unlink()

    scipy.io.savemat(
        str(input_path),
        {
            "calcium_matrix": np.asarray(traces, dtype=np.float64),
            "roi_ids": np.asarray(roi_ids, dtype=np.float64),
            "fps": float(fps),
            "params": params,
        },
    )

    matlab_call = (
        f"run_model_batch_from_files('{_to_matlab_path(input_path)}', "
        f"'{_to_matlab_path(output_path)}', '{model_name}')"
    )
    proc = _run_matlab(matlab_executable, matlab_call, log_prefix="matlab batch")

    if proc.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"MATLAB batch model run failed (model={model_name!r}, returncode={proc.returncode}). "
            f"See stdout/stderr above. Input file left at {input_path} for inspection."
        )

    mat = scipy.io.loadmat(str(output_path))
    predicted_events = np.asarray(mat["predicted_events"]).astype(bool)
    n_predicted_events = np.asarray(mat["n_predicted_events"]).astype(int).ravel()
    failed_roi_ids = [int(x) for x in np.asarray(mat.get("failed_roi_ids", [])).ravel()]
    failed_roi_reasons = _decode_matlab_cellstr(mat.get("failed_roi_reasons"))

    if failed_roi_ids:
        print(f"[matlab batch] WARNING: {len(failed_roi_ids)} ROI(s) failed and were "
              f"recorded as 0 events (not dropped, not counted as real negatives):")
        for roi_id, reason in zip(failed_roi_ids, failed_roi_reasons):
            print(f"  #{roi_id}: {reason}")

    return {
        "predicted_events": predicted_events,
        "n_predicted_events": n_predicted_events,
        "failed_roi_ids": failed_roi_ids,
        "failed_roi_reasons": failed_roi_reasons,
    }


def _decode_matlab_cellstr(arr) -> list[str]:
    """scipy.io.loadmat represents a MATLAB cell array of strings as a
    nested object array; this pulls the plain Python strings back out."""
    if arr is None:
        return []
    out = []
    for item in np.asarray(arr).ravel():
        item = np.asarray(item).ravel()
        out.append(str(item[0]) if len(item) else "")
    return out


def save_predicted_events(
    rois: list[Roi], predicted_events: np.ndarray, sampling_rate_hz: float, out_dir: Path
) -> None:
    """
    Saves per-neuron predicted events to disk in two forms, keyed by ROI id
    (not row index) so downstream stages can't misalign after any reordering:
      - a wide .npz (predicted_events bool matrix + roi_ids + sampling_rate_hz)
      - a long-format CSV (roi_id, event_time_sec) for quick inspection / the report
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_ids = np.array([r.id for r in rois])

    np.savez(
        out_dir / "predicted_events.npz",
        roi_ids=roi_ids,
        predicted_events=predicted_events,
        sampling_rate_hz=sampling_rate_hz,
    )

    csv_path = out_dir / "predicted_events.csv"
    with open(csv_path, "w") as f:
        f.write("roi_id,event_time_sec\n")
        for roi, events in zip(rois, predicted_events):
            for frame_idx in np.where(events)[0]:
                f.write(f"{roi.id},{frame_idx / sampling_rate_hz:.4f}\n")


def save_pipeline_cache(
    rois: list[Roi], traces: np.ndarray, predicted_events: np.ndarray,
    sampling_rate_hz: float, cache_path: Path,
) -> None:
    """
    Cheap-to-store results (rois/traces/events -- kilobytes, not the ~2GB
    dff_stack) so `--gui-only` can redraw the GUI without repeating
    detection/extraction/the MATLAB model run.

    Rois are stored as plain dicts, not pickled Roi instances: a pickled
    dataclass instance embeds a reference to its class's module + qualname,
    which resolves to "__main__" when this file is run directly (the normal
    way) -- fine as long as loading also happens via `python
    neuron_trace_pipeline.py`, but it breaks the moment anything loads the
    cache after importing this file as a module instead (hit exactly this
    while writing a standalone test for this feature). Plain dicts have no
    such dependency.
    """
    import pickle
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    roi_dicts = [
        {"id": r.id, "bbox": r.bbox, "centroid": r.centroid, "area_px": r.area_px, "coords": r.coords}
        for r in rois
    ]
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"rois": roi_dicts, "traces": traces, "predicted_events": predicted_events,
             "sampling_rate_hz": sampling_rate_hz},
            f,
        )


def load_pipeline_cache(cache_path: Path) -> dict:
    import pickle
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No pipeline cache at {cache_path}. Run the script normally first "
            "(python neuron_trace_pipeline.py, no arguments) to produce one -- "
            "--gui-only only redraws the GUI from a previous full run's results."
        )
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    data["rois"] = [Roi(**d) for d in data["rois"]]
    return data


def plot_single_roi_model_result(
    calcium: np.ndarray, result: dict, sampling_rate_hz: float, roi_id: int,
    model_name: str, out_path: Path,
) -> None:
    """
    Plots one trace against its model result (pairs with run_model_via_matlab()).
    Not called from main()'s default flow -- kept as a lower-level utility
    for ad-hoc single-trace debugging.
    """
    # See draw_detection_overlay() -- backend is set centrally, not here.
    import matplotlib.pyplot as plt

    t = np.arange(len(calcium)) / sampling_rate_hz
    predicted_events = np.asarray(result["predicted_events"]).astype(bool).ravel()
    event_times = t[predicted_events]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, calcium, color="tab:blue", linewidth=1.0, label="ΔF/F (calcium)")
    for i, et in enumerate(event_times):
        ax.axvline(et, color="tab:red", linestyle="--", linewidth=0.8,
                    label="predicted event" if i == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔF/F")
    ax.set_title(f"Neuron #{roi_id} — {model_name} predicted events (n={predicted_events.sum()})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Stage 4a: static GUI mockup (this checkpoint). Video-with-boxes on the
# right, stacked traces-with-event-markers on the left. A live/animated
# version (Stage 4b) comes after this is confirmed to look right.
# =============================================================================

def build_static_gui_mockup(
    display_image: np.ndarray,
    rois: list[Roi],
    traces: np.ndarray,
    predicted_events: np.ndarray,
    sampling_rate_hz: float,
    out_path: Path,
) -> None:
    """
    Two-panel static preview of the eventual live GUI layout:
      left  = traces stacked bottom(#1)-to-top, ΔF/F(t), with predicted
              event markers (small ticks) on each neuron's own row
      right = one representative frame with every ROI boxed and labeled

    Color-coding rule (per spec): unique per-neuron color for both box and
    trace only when neuron count stays visually distinguishable (<=12);
    above that, neutral/uniform style with number labels only. This video
    has 20 ROIs, so both panels intentionally use a single neutral color.
    """
    # See draw_detection_overlay() -- backend is set centrally, not here.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_rois, _n_frames = traces.shape
    color_coded = n_rois <= 12
    cmap = plt.get_cmap("tab20" if n_rois > 10 else "tab10")
    neutral_trace_color = "steelblue"
    neutral_box_color = "yellow"

    fig, (ax_traces, ax_video) = plt.subplots(1, 2, figsize=(16, max(6, 0.4 * n_rois)))

    # --- Left: stacked traces with event ticks ---
    _draw_stacked_traces(ax_traces, traces, rois, sampling_rate_hz, color_coded, cmap,
                          neutral_trace_color, predicted_events=predicted_events)
    ax_traces.set_xlabel("time (s)")
    ax_traces.set_title("ΔF/F(t) per neuron, predicted events marked (▼)")

    # --- Right: representative frame with boxed, labeled ROIs ---
    p_lo, p_hi = np.percentile(display_image, (1, 99.5))
    ax_video.imshow(display_image, cmap="gray", vmin=p_lo, vmax=p_hi)
    for i, roi in enumerate(rois):
        color = cmap(i % cmap.N) if color_coded else neutral_box_color
        min_row, min_col, max_row, max_col = roi.bbox
        ax_video.add_patch(Rectangle(
            (min_col, min_row), max_col - min_col, max_row - min_row,
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        ax_video.text(min_col, min_row - 3, f"#{roi.id}", color=color,
                       fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax_video.set_title(f"Detected neurons (n={n_rois}) -- static frame")
    ax_video.axis("off")

    fig.suptitle("Stage 4 static mockup (video will animate; traces will scroll live)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def launch_gui(
    stack: np.ndarray,
    rois: list[Roi],
    traces: np.ndarray,
    predicted_events: np.ndarray,
    sampling_rate_hz: float,
    playback_mode: str,
    playback_speed: float,
    video_output_path: Path | None = None,
    video_max_frames: int | None = None,
    video_dpi: int = 100,
    snapshot_path: Path | None = None,
) -> None:
    """
    Live/animated version of the Stage 4a static mockup: the right panel
    plays the RAW video frame-by-frame (not ΔF/F -- ΔF/F values hover near 0
    with little contrast, which is why an earlier version of this looked
    washed out/blurry; the raw stack looks like the actual recording) with
    the (fixed-position) detection boxes overlaid; the left panel holds the
    already-known traces + event markers static, with a moving time cursor
    synced to the video frame.

    In "interactive" mode there's also a scrub Slider (drag to any frame,
    in either direction) and a Play/Pause Button, so playback isn't just a
    passive loop. "save_video" mode has no interactive widgets (nothing to
    interact with) -- it stays a passive render for sharing, saved as an
    actual .mp4 (via OpenCV's mp4v encoder -- no ffmpeg install needed, and
    it compresses this grainy grayscale content far better than the GIF
    approach we started with did). "interactive" mode ALSO always leaves
    files on disk, not just a live window: a PNG snapshot of whatever frame
    was showing when you close the window (snapshot_path), and, once
    closed, the same .mp4 that save_video mode produces (video_output_path)
    -- so every run leaves a shareable artifact behind.

    Implementation choice: matplotlib (FuncAnimation + widgets.Slider/Button),
    not PyQt. Justification (per spec: "choose whichever is more reliable to
    get working correctly by the deadline"): matplotlib is already the only
    plotting dependency used everywhere else in this script and in the
    existing MATLAB-side plots; PyQt would be a new dependency with its own
    event-loop/threading concerns, for scrub/play/pause controls that
    matplotlib.widgets already covers natively. blit=False is used
    deliberately: with two axes (image + line plot) updating together,
    blitting is fragile across backends/OSes, and the reliability trade-off
    isn't worth the extra speed for this deliverable.
    """
    # Backend is set centrally in _configure_matplotlib_backend(), called at
    # the top of main()/gui_only_main() -- BEFORE any of the earlier
    # plotting functions run. Forcing Agg here (or in those earlier
    # functions) after the fact is exactly what broke plt.show() before:
    # once a backend is picked and a canvas exists, matplotlib can't
    # reliably switch to a different one later in the same process.
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Slider, Button
    import cv2

    n_rois, n_frames = traces.shape
    t = np.arange(n_frames) / sampling_rate_hz
    color_coded = n_rois <= 12
    cmap = plt.get_cmap("tab20" if n_rois > 10 else "tab10")
    neutral_trace_color = "steelblue"
    neutral_box_color = "yellow"

    figsize = (16, max(6, 0.4 * n_rois))
    fig, (ax_traces, ax_video) = plt.subplots(1, 2, figsize=figsize)
    if playback_mode == "interactive":
        fig.subplots_adjust(bottom=0.18)  # room for the slider/button row

    # --- Left: static traces + event markers + moving time cursor ---
    _draw_stacked_traces(ax_traces, traces, rois, sampling_rate_hz, color_coded, cmap,
                          neutral_trace_color, predicted_events=predicted_events)
    cursor_line = ax_traces.axvline(t[0], color="black", linewidth=1.2, zorder=4)
    ax_traces.set_xlabel("time (s)")
    ax_traces.set_title("ΔF/F(t) per neuron, predicted events (▼), live cursor")

    # --- Right: animated RAW video frame + static boxes/labels ---
    vmin, vmax = np.percentile(stack, (1, 99.5))
    im = ax_video.imshow(stack[0], cmap="gray", vmin=vmin, vmax=vmax)
    for i, roi in enumerate(rois):
        color = cmap(i % cmap.N) if color_coded else neutral_box_color
        min_row, min_col, max_row, max_col = roi.bbox
        ax_video.add_patch(Rectangle(
            (min_col, min_row), max_col - min_col, max_row - min_row,
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        ax_video.text(min_col, min_row - 3, f"#{roi.id}", color=color,
                       fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax_video.axis("off")
    frame_title = ax_video.set_title("")

    fig.suptitle(f"Live view -- {n_rois} neurons, {sampling_rate_hz:.1f}Hz, "
                 f"{playback_speed:.1f}x speed")
    if playback_mode != "interactive":
        fig.tight_layout()

    def set_frame(frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        im.set_data(stack[frame_idx])
        cursor_line.set_xdata([t[frame_idx], t[frame_idx]])
        frame_title.set_text(f"t = {t[frame_idx]:.2f}s (frame {frame_idx + 1}/{n_frames})")

    if playback_mode == "interactive":
        # --- Scrub bar + Play/Pause, single source of truth for "current frame" ---
        ax_slider = fig.add_axes([0.15, 0.05, 0.55, 0.03])
        frame_slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

        ax_play_button = fig.add_axes([0.75, 0.035, 0.1, 0.05])
        play_button = Button(ax_play_button, "Play")

        state = {"playing": False}

        def on_slider_changed(val):
            set_frame(val)
            fig.canvas.draw_idle()

        frame_slider.on_changed(on_slider_changed)

        def toggle_play(_event):
            state["playing"] = not state["playing"]
            play_button.label.set_text("Pause" if state["playing"] else "Play")

        play_button.on_clicked(toggle_play)

        def advance(_frame):
            if state["playing"]:
                next_idx = (int(frame_slider.val) + 1) % n_frames
                frame_slider.set_val(next_idx)  # triggers on_slider_changed -> set_frame
            return ()

        interval_ms = 1000.0 / sampling_rate_hz / playback_speed
        ani = animation.FuncAnimation(
            fig, advance, interval=interval_ms, blit=False, cache_frame_data=False,
        )

        if snapshot_path is not None:
            def on_close(_event):
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(snapshot_path), dpi=120)
                print(f"[gui] snapshot (frame {int(frame_slider.val) + 1}/{n_frames}) "
                      f"saved to {snapshot_path}")
            fig.canvas.mpl_connect("close_event", on_close)

        plt.show()  # blocks until the window is closed

        if video_output_path is not None:
            print("[gui] window closed -- also exporting the .mp4 (same as save_video mode)...")
            launch_gui(
                stack=stack, rois=rois, traces=traces, predicted_events=predicted_events,
                sampling_rate_hz=sampling_rate_hz, playback_mode="save_video",
                playback_speed=playback_speed, video_output_path=video_output_path,
                video_max_frames=video_max_frames, video_dpi=video_dpi,
            )

    elif playback_mode == "save_video":
        if video_output_path is None:
            raise ValueError("video_output_path is required when playback_mode='save_video'")

        # None/too-large max_frames -> every frame, full temporal
        # resolution (mp4 compresses this well enough that we don't need
        # GIF's harsh subsampling). Otherwise subsample evenly across the
        # FULL time range so the preview still covers the whole clip.
        if video_max_frames is not None and video_max_frames < n_frames:
            frame_indices = np.unique(np.linspace(0, n_frames - 1, video_max_frames).astype(int))
        else:
            frame_indices = np.arange(n_frames)

        fig.set_dpi(video_dpi)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        video_fps = sampling_rate_hz * playback_speed * len(frame_indices) / n_frames

        set_frame(frame_indices[0])
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_output_path), fourcc, video_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open for {video_output_path} -- codec 'mp4v' "
                "may not be available in this OpenCV build."
            )
        try:
            for frame_idx in frame_indices:
                set_frame(frame_idx)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
        plt.close(fig)
    else:
        raise ValueError(f"Unknown GUI_PLAYBACK_MODE: {playback_mode!r}")


def _configure_matplotlib_backend(playback_mode: str) -> None:
    """
    Picks the matplotlib backend ONCE, before any figure is created anywhere
    in this script. Must run before draw_detection_overlay/plot_example_traces/
    plot_single_roi_model_result/build_static_gui_mockup/launch_gui -- calling
    matplotlib.use() again after a canvas already exists doesn't reliably
    switch backends (this was a real bug: earlier plotting functions each
    forced Agg, which silently broke launch_gui()'s later plt.show()).

    playback_mode="save_video" doesn't need a display at all, so Agg (headless,
    no extra dependencies) is used deliberately. playback_mode="interactive"
    needs a GUI-capable backend; TkAgg is tried first (ships with a normal
    CPython install's "tcl/tk and IDLE" component), then Qt as a fallback.
    """
    import matplotlib

    if playback_mode != "interactive":
        matplotlib.use("Agg")
        return

    last_error = None
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend)
            print(f"[gui] matplotlib backend: {backend}")
            return
        except Exception as e:
            last_error = e
            print(f"[gui] backend {backend!r} unavailable ({e}); trying next...")

    raise RuntimeError(
        "No interactive matplotlib backend is available in this Python "
        "environment (tried TkAgg, QtAgg, Qt5Agg). Either:\n"
        "  (a) pip install PyQt5   -- inside THIS venv/interpreter, then rerun; or\n"
        "  (b) make sure Tk is available: run `python -c \"import tkinter\"` in "
        "this venv -- if that fails, the Python this venv was created from is "
        "missing Tcl/Tk (on Windows: rerun the python.org installer, choose "
        "Modify, and enable the 'tcl/tk and IDLE' feature, then recreate the venv); or\n"
        "  (c) set CONFIG['GUI_PLAYBACK_MODE'] = 'save_video' to get an .mp4 "
        "instead of a live window.\n"
        f"Last import error: {last_error}"
    )


# =============================================================================
# Main
# =============================================================================

def _make_run_dir(output_dir: Path) -> Path:
    """New timestamped subfolder for one full run: OUTPUT_DIR/YYYYMMDD_HHMMSS/."""
    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_latest_run_dir(output_dir: Path) -> Path:
    """
    Most recent OUTPUT_DIR/YYYYMMDD_HHMMSS/ subfolder that actually has a
    pipeline_cache.pkl in it -- used by --gui-only, which needs a previous
    full run's results but isn't told which one. Timestamp-named folders
    sort correctly as plain strings, so the lexicographically last one is
    also the most recent.
    """
    if not output_dir.exists():
        raise FileNotFoundError(
            f"{output_dir} doesn't exist -- run the script normally first "
            "(python neuron_trace_pipeline.py, no arguments) to produce a run folder."
        )
    candidates = sorted(
        p for p in output_dir.iterdir() if p.is_dir() and (p / PIPELINE_CACHE_FILENAME).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run folder with a {PIPELINE_CACHE_FILENAME} found under {output_dir} -- "
            "run the script normally first (python neuron_trace_pipeline.py, no arguments)."
        )
    return candidates[-1]


def main() -> None:
    cfg = CONFIG
    _configure_matplotlib_backend(cfg["GUI_PLAYBACK_MODE"])
    out_dir = _make_run_dir(Path(cfg["OUTPUT_DIR"]))
    print(f"[run] output folder: {out_dir}")

    print(f"[load] {cfg['VIDEO_PATH']}")
    stack, tf = load_stack(cfg["VIDEO_PATH"])
    print(f"[load] stack shape (T, Y, X) = {stack.shape}")

    measured_hz = check_frame_rate(tf, cfg["SAMPLING_RATE_HZ"], cfg["FPS_MISMATCH_TOLERANCE"])
    effective_hz = measured_hz if measured_hz is not None else cfg["SAMPLING_RATE_HZ"]

    f0 = compute_f0(stack)
    dff_stack = compute_dff_stack(stack, f0)

    rois, binary_mask, max_proj = detect_neurons(
        dff_stack,
        threshold_method=cfg["THRESHOLD_METHOD"],
        min_roi_size_px=cfg["MIN_ROI_SIZE_PX"],
        max_roi_size_px=cfg["MAX_ROI_SIZE_PX"],
        spatial_smooth_sigma_px=cfg["SPATIAL_SMOOTH_SIGMA_PX"],
    )
    print(f"[detect] found {len(rois)} ROIs "
          f"(threshold_method={cfg['THRESHOLD_METHOD']}, "
          f"size_range=[{cfg['MIN_ROI_SIZE_PX']}, {cfg['MAX_ROI_SIZE_PX']}]px)")
    if len(rois) == 0:
        raise RuntimeError(
            "No ROIs detected -- nothing to extract traces from. Check "
            "THRESHOLD_METHOD/MIN_ROI_SIZE_PX/MAX_ROI_SIZE_PX/SPATIAL_SMOOTH_SIGMA_PX "
            "in CONFIG against this video (a new video's brightness/contrast can need "
            "different values than the one these defaults were tuned on)."
        )
    for roi in rois:
        print(f"  #{roi.id}: bbox={roi.bbox}, centroid=({roi.centroid[0]:.1f}, {roi.centroid[1]:.1f}), "
              f"area={roi.area_px}px")

    overlay_path = out_dir / "detected_neurons_overlay.png"
    draw_detection_overlay(max_proj, rois, overlay_path)
    print(f"[detect] overlay saved to {overlay_path}")

    traces = extract_traces(dff_stack, rois)
    print(f"\n[extract] traces shape (n_rois, T) = {traces.shape}")
    for roi, trace in zip(rois, traces):
        print(f"  #{roi.id}: dF/F min={trace.min():.3f} max={trace.max():.3f} "
              f"mean={trace.mean():.3f} std={trace.std():.3f}")

    traces_path = out_dir / "example_traces.png"
    plot_example_traces(traces, rois, effective_hz, traces_path)
    print(f"[extract] example traces plot saved to {traces_path}")

    # NEURON_SELECTION_MODE governs what reaches the model and GUI from here
    # on (Stage 1/2 outputs above stay a full census of everything detected,
    # which is what makes them useful as detection/extraction diagnostics).
    # With mode="all" (the default), this is a no-op: select_neurons()
    # returns every index in order, so selected_rois/selected_traces are
    # exactly rois/traces.
    selected = select_neurons(rois, traces, cfg["NEURON_SELECTION_MODE"], cfg["TOP_K"])
    if len(selected) == 0:
        raise RuntimeError(
            f"NEURON_SELECTION_MODE={cfg['NEURON_SELECTION_MODE']!r} with "
            f"TOP_K={cfg['TOP_K']!r} selected 0 neurons -- nothing left for the model/GUI stages."
        )
    selected_rois = [rois[i] for i in selected]
    selected_traces = traces[selected]
    print(f"\n[select] mode={cfg['NEURON_SELECTION_MODE']!r} -> "
          f"{len(selected_rois)}/{len(rois)} neurons selected for model+GUI: "
          f"{[r.id for r in selected_rois]}")

    # --- Stage 3: model, all selected ROIs in one MATLAB session ---
    print(f"\n=== Stage 3 (MATLAB model run): model={cfg['MODEL_NAME']!r} ===")
    model_params = cfg["MODEL_PARAMS"][cfg["MODEL_NAME"]]
    matlab_model_name = MODEL_DISPATCH[cfg["MODEL_NAME"]]
    io_dir = out_dir / "matlab_io"
    batch_result = run_model_batch_via_matlab(
        traces=selected_traces,
        roi_ids=[r.id for r in selected_rois],
        fps=effective_hz,
        model_name=matlab_model_name,
        params=model_params,
        io_dir=io_dir,
        matlab_executable=cfg["MATLAB_EXECUTABLE"],
    )
    predicted_events = batch_result["predicted_events"]
    for roi, n_ev in zip(selected_rois, batch_result["n_predicted_events"]):
        print(f"  #{roi.id}: {n_ev} predicted events")

    events_dir = out_dir / "events"
    save_predicted_events(selected_rois, predicted_events, effective_hz, events_dir)
    print(f"[model] predicted events saved to {events_dir}\\predicted_events.npz "
          f"and {events_dir}\\predicted_events.csv")

    cache_path = out_dir / PIPELINE_CACHE_FILENAME
    save_pipeline_cache(selected_rois, selected_traces, predicted_events, effective_hz, cache_path)
    print(f"[cache] rois/traces/predicted_events cached to {cache_path} "
          f"(use --gui-only to redraw Stage 4b without rerunning stages 1-3)")

    # --- Stage 4a: static GUI mockup ---
    mockup_path = out_dir / "stage4_static_gui_mockup.png"
    build_static_gui_mockup(max_proj, selected_rois, selected_traces, predicted_events,
                             effective_hz, mockup_path)
    print(f"[gui] static mockup saved to {mockup_path}")

    # --- Stage 4b: live/animated GUI ---
    video_output_path = out_dir / cfg["GUI_VIDEO_FILENAME"]
    snapshot_path = out_dir / cfg["GUI_SNAPSHOT_FILENAME"]
    print(f"\n=== Stage 4b (live GUI): mode={cfg['GUI_PLAYBACK_MODE']!r} ===")
    launch_gui(
        stack=stack,
        rois=selected_rois,
        traces=selected_traces,
        predicted_events=predicted_events,
        sampling_rate_hz=effective_hz,
        playback_mode=cfg["GUI_PLAYBACK_MODE"],
        playback_speed=cfg["GUI_PLAYBACK_SPEED"],
        video_output_path=video_output_path,
        video_max_frames=cfg["GUI_VIDEO_MAX_FRAMES"],
        video_dpi=cfg["GUI_VIDEO_DPI"],
        snapshot_path=snapshot_path,
    )
    if cfg["GUI_PLAYBACK_MODE"] == "save_video":
        print(f"[gui] live demo video saved to {video_output_path}")

    print("\n--- Stage 4b complete. This covers the full spec: video -> detection -> "
          "traces -> model -> live GUI. ---")
    print(f"--- Everything from this run is under: {out_dir} ---")


def gui_only_main() -> None:
    """
    `python neuron_trace_pipeline.py --gui-only` -- redraws Stage 4b from a
    cached previous run's rois/traces/predicted_events instead of repeating
    detection (Stage 1), extraction (Stage 2), and the MATLAB model run
    (Stage 3, the slowest part). Still reloads the video and recomputes
    dff_stack (fast: plain numpy, no MATLAB) since that's the ~2GB video
    background the GUI plays -- not worth caching to disk.
    """
    cfg = CONFIG
    _configure_matplotlib_backend(cfg["GUI_PLAYBACK_MODE"])

    run_dir = _find_latest_run_dir(Path(cfg["OUTPUT_DIR"]))
    print(f"[run] using most recent run folder: {run_dir}")

    cache_path = run_dir / PIPELINE_CACHE_FILENAME
    cache = load_pipeline_cache(cache_path)
    print(f"[cache] loaded rois/traces/predicted_events from {cache_path}")

    print(f"[load] {cfg['VIDEO_PATH']}")
    stack, _tf = load_stack(cfg["VIDEO_PATH"])

    video_output_path = run_dir / cfg["GUI_VIDEO_FILENAME"]
    snapshot_path = run_dir / cfg["GUI_SNAPSHOT_FILENAME"]
    print(f"\n=== Stage 4b (live GUI, --gui-only): mode={cfg['GUI_PLAYBACK_MODE']!r} ===")
    launch_gui(
        stack=stack,
        rois=cache["rois"],
        traces=cache["traces"],
        predicted_events=cache["predicted_events"],
        sampling_rate_hz=cache["sampling_rate_hz"],
        playback_mode=cfg["GUI_PLAYBACK_MODE"],
        playback_speed=cfg["GUI_PLAYBACK_SPEED"],
        video_output_path=video_output_path,
        video_max_frames=cfg["GUI_VIDEO_MAX_FRAMES"],
        video_dpi=cfg["GUI_VIDEO_DPI"],
        snapshot_path=snapshot_path,
    )
    if cfg["GUI_PLAYBACK_MODE"] == "save_video":
        print(f"[gui] live demo video saved to {video_output_path}")


if __name__ == "__main__":
    if "--gui-only" in sys.argv:
        gui_only_main()
    else:
        main()
