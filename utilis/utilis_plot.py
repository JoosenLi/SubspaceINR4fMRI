import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

def make_block_events_variable(n_scans, tr, block_durations, rest_duration=30.0):
    """
    Generate block-design events for fMRI, allowing variable block durations.
    - block_durations: list or array of durations for each block (in seconds)
    """
    total_time = n_scans * tr
    onsets = []
    current_onset = 0.0

    # Generate onsets block by block
    for i, dur in enumerate(block_durations):
        if current_onset + dur <= total_time:
            onsets.append(current_onset)
            current_onset += dur + rest_duration
        else:
            break  # stop if next block exceeds total scan duration

    events = pd.DataFrame({
        'onset': onsets,
        'duration': block_durations[:len(onsets)],
        'trial_type': ['checkerboard'] * len(onsets)
    })

    # Diagnostics
    print(f"[make_block_events_variable] total_time={total_time:.1f}s, n_blocks={len(onsets)}")
    if len(onsets) > 0:
        print(f" first onset={onsets[0]:.3f}s, last onset={onsets[-1]:.3f}s, "
              f"last block end={onsets[-1] + block_durations[len(onsets)-1]:.3f}s")

    return events


def run_first_level_single_session(fmri_img, motion_params, n_scans, tr,
                                   block_duration=30, rest_duration=30.0):
    """
    Full single-session first-level GLM pipeline for block design.
    Returns (z_map, design_matrix, glm, events).
    """
    # Frame times
    frame_times = np.arange(n_scans) * tr

    # Generate events
    block_durations = [30, 30, 30, 30]
    rest_duration = 30.0

    events = make_block_events_variable(n_scans, tr, block_durations, rest_duration)

    # Motion parameters check
    add_regs = np.asarray(motion_params)
    if add_regs.ndim != 2:
        raise ValueError("motion_params must have shape (n_scans, n_params)")
    if add_regs.shape[0] != n_scans:
        raise ValueError(f"motion_params rows ({add_regs.shape[0]}) must match n_scans ({n_scans})")

    add_reg_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']

    # Design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model='spm',
        drift_model='polynomial',
        drift_order=5,
        add_regs=add_regs,
        add_reg_names=add_reg_names
    )

    # Fit GLM
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model='spm',
        drift_model='polynomial',
        drift_order=5
    )
    glm = glm.fit(fmri_img, design_matrices=design_matrix)

    # Contrast: checkerboard > baseline
    if 'checkerboard' not in design_matrix.columns:
        raise ValueError("No 'checkerboard' column found in design matrix. Check events.")
    contrast_vec = (design_matrix.columns == 'checkerboard').astype(float)
    print(contrast_vec)
    res = glm.compute_contrast(contrast_vec, output_type='all')
    z_map = res['z_score']

    return z_map, design_matrix, glm, events

import matplotlib.gridspec as gridspec
def check_model_fit_at_peak(fmri_img, z_map, design_matrix, glm,
                            reg_idx=0, title='Model fit', plot=True):
    """
    fmri_img: 4D Niimg for one reconstruction (e.g. R=1 或 R=4)
    z_map: Niimg z-map from GLM (same space / masker as glm)
    design_matrix: pandas DataFrame (n_scans x n_regressors)
    glm: fitted nilearn FirstLevelModel (已经 fit 好)
    reg_idx: index of the regressor to consider as "first" (默认 0)
    """
    # ---- 1) extract masker and data ----
    masker = glm.masker_  # nilearn masker used by the GLM
    # z values in masker space (shape (1, n_voxels))
    z_vals = masker.transform(z_map).ravel()
    if np.all(np.isnan(z_vals)):
        raise RuntimeError("z_map yields only NaNs through glm.masker_. Check masks / alignment.")
    peak_idx = np.nanargmax(z_vals)
    print(z_vals[peak_idx])
    # Extract voxel time series from fmri_img (shape: n_scans x n_voxels)
    data_ts = masker.transform(fmri_img)  # -> (n_scans, n_voxels)

    voxel_ts = data_ts[:, peak_idx]
    n_scans = voxel_ts.shape[0]
    # Sanity check
    # ---- 2) prepare design matrix ----
    X = design_matrix.values
    if X.shape[0] != n_scans:
        raise ValueError(f"Design matrix rows ({X.shape[0]}) != n_scans ({n_scans}).")
    # optionally z-score or not — here we use raw design matrix (consistent with GLM fit)
    
    # ---- 3) estimate betas at this voxel by ordinary least squares ----
    # pinv for numerical stability
    betas = np.linalg.pinv(X) @ voxel_ts  # shape (n_regressors,)
    predicted_full = X @ betas
    predicted_first = X[:, reg_idx] * betas[reg_idx]
    residuals = voxel_ts - predicted_full

    # Diagnostics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((voxel_ts - voxel_ts.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))

    if plot:
        t = np.arange(n_scans)
        # --- 设置绘图布局：上下两部分 ---
        fig = plt.figure(figsize=(10, 6))
        # height_ratios 可以调整上下图的高度比例
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
        
        ax1 = plt.subplot(gs[0]) # 上半部分：显示高数值 (Observed, Predicted)
        ax2 = plt.subplot(gs[1], sharex=ax1) # 下半部分：显示低数值 (Reg 1)

        # --- 1. 绘制上半部分 (Observed & Predicted) ---
        ax1.plot(t, voxel_ts, label='Observed BOLD', linewidth=1.5, color='#1f77b4') # Blue
        ax1.plot(t, predicted_full, label='Predicted (all regs)', linewidth=1.5, color='#ff7f0e') # Orange
        
        # 动态设置上半部分的 Y 轴范围 (聚焦在信号附近)
        y_high_data = np.concatenate([voxel_ts, predicted_full])
        margin_high = (y_high_data.max() - y_high_data.min()) * 0.2
        ax1.set_ylim(y_high_data.min() - margin_high, y_high_data.max() + margin_high)
        
        ax1.legend(loc='upper right')
        ax1.set_title(f"{title} (Broken Axis View)")

        # --- 2. 绘制下半部分 (Regressor contribution) ---
        ax2.plot(t, predicted_first, label=f'Contribution of reg {reg_idx+1}', 
                 linestyle='--', color='#2ca02c', linewidth=1.5) # Green
        
        # 动态设置下半部分的 Y 轴范围 (聚焦在回归量附近)
        margin_low = (predicted_first.max() - predicted_first.min()) * 0.2
        if margin_low == 0: margin_low = 0.0001 # 防止平线报错
        ax2.set_ylim(predicted_first.min() - margin_low, predicted_first.max() + margin_low)
        
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Time (frames)')
        ax2.set_ylabel('Signal (a.u.)', y=1.0) # 将 Y 轴标签放在中间位置

        # --- 3. 美化：隐藏中间的轴脊，制造断裂效果 ---
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # 移除上图的 x 轴标签
        ax2.xaxis.tick_bottom()

        # 添加断裂线 (d)
        d = .015  
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        # 上图的断裂线
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # 切换到下图坐标系
        # 下图的断裂线
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        plt.subplots_adjust(hspace=0.1) # 调整子图间距
        plt.show()

        # Residuals plot
        plt.figure(figsize=(10,2.5))
        plt.plot(t, residuals, label='Residuals', linewidth=1)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.xlabel('Time (frames)')
        plt.title('Residuals')
        plt.show()

        # Optional: scatter observed vs predicted
        plt.figure(figsize=(4,4))
        plt.scatter(predicted_full, voxel_ts, s=8)
        plt.xlabel('Predicted (all regs)')
        plt.ylabel('Observed')
        plt.title('Observed vs Predicted')
        plt.show()

    return {
        'peak_idx': int(peak_idx),
        'betas': betas,
        'predicted_full': predicted_full,
        'predicted_first': predicted_first,
        'residuals': residuals,
        'r2': r2,
        'rmse': rmse
    }


def plot_slices_pro(volume, slices=None, title="Orthogonal Slices", cmap='gray'):
    """
    Jupyter Notebook 优化的 3D 切片绘图函数。
    修复了长宽比失真、Colorbar 重叠和布局拥挤的问题。
    """
    
    if volume.ndim != 3:
        raise ValueError("数据必须是 3D 的")
    
    # 1. 智能计算切片位置 (如果未提供)
    dims = volume.shape
    if slices is None:
        slices = [d // 2 for d in dims]
    x_idx, y_idx, z_idx = slices
    
    # 提取切片
    # 注意：根据你的图片，似乎需要旋转一下才能正对着看 (通常 MRI 数据需要 rot90)
    # 这里我保持原始数据方向，若方向不对，可以在提取时加 np.rot90()
    slice_0 = volume[x_idx, :, :] 
    slice_1 = volume[:, y_idx, :].T 
    slice_2 = volume[:, :, z_idx]

    # 2. 创建画布 - 关键点解释：
    # figsize=(12, 5): 增加宽度，防止子图挤在一起
    # constrained_layout=True: 这是 matplotlib 较新的功能，比 tight_layout 更智能，
    # 它可以自动处理 Colorbar 与子图之间的空隙，防止重叠。
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    
    # 设置总标题
    fig.suptitle(f"{title} | Shape: {dims} | Slice Idx: {slices}", fontsize=16, fontweight='bold')

    # 3. 绘图配置列表
    # 我们把要画的数据和对应的标签组织起来，用循环处理，代码更整洁
    plot_data = [
        (slice_0, f"Sagittal (X={x_idx})", "Axis 1", "Axis 2"), # 侧面
        (slice_1, f"Coronal (Y={y_idx})",  "Axis 0", "Axis 2"), # 正面
        (slice_2, f"Axial (Z={z_idx})",    "Axis 0", "Axis 1")  # 顶面
    ]

    images = [] # 存储 image 对象以便后续画 colorbar

    for ax, (data, ax_title, ylabel, xlabel) in zip(axes, plot_data):
        # --- 核心修复 ---
        # aspect='equal': 强制像素显示为正方形。这会修复“压扁”的问题，
        # 虽然这会导致 64x40 的图比 64x64 的图看起来矮，但这是物理上正确的。
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='equal', interpolation='nearest')
        
        ax.set_title(ax_title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        # 去掉刻度线（可选，看起来更干净）
        # ax.axis('off') 
        images.append(im)

    # 4. 更加美观的 Colorbar
    # 我们不让 Colorbar 抢占某个子图的位置，而是让它附属于整个 Figure
    # aspect=30 控制 colorbar 的细长程度
    cbar = fig.colorbar(images[-1], ax=axes, orientation='vertical', fraction=0.05, aspect=30)
    cbar.set_label('Intensity', fontsize=10)

    plt.show()



def plot_activation_2d(z_score, threshold, rec_zer, roi_resampled, z_thresh):
    from snake.toolkit.plotting import plot_frames_activ
    from mpl_toolkits.axes_grid1.axes_divider import Size, make_axes_locatable
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(3, 3))
    roi_thresh= roi_resampled > threshold 
    plot_frames_activ(
        background=abs(rec_zer[0]).T,
        z_score=z_score.T,
        rois=[roi_thresh.T],
        ax=ax,
        slices=None, 
        bbox=None, 
        z_thresh=z_thresh,  
        z_max=8,  
        bg_cmap="gray",  
        roi_colors=["cyan"],
    )
    plt.tight_layout()

def plot_activation_3d(rec_data, z_score, threshold, width_inches=7, vmin_vmax=None, cbar=True, cuts=None, n_cols=1, roi_resampled=None, thresh=None):
    from snake.toolkit.plotting import axis3dcut,get_mask_cuts_mask,_get_axis_properties
    roi_thresh= roi_resampled > threshold 
    roi_cuts = get_mask_cuts_mask(roi_thresh)
    # hdiv, vdiv, bbox, slices = _get_axis_properties(
    #     rec_data[0], roi_cuts, width_inches, cbar=cbar
    # )
    fig, ax, cuts= axis3dcut(background=abs(rec_data[-1]).T, z_score=z_score, gt_roi=roi_thresh.T, vmin_vmax =  vmin_vmax, cuts=cuts, width_inches=width_inches, cbar=cbar, roi_colors=['cyan'],z_thresh=thresh)
    print(cuts)
    return fig, ax, cuts

def cluster_threshold_3d(
    act: np.ndarray,
    voxel_thresh: float = 0.0,
    min_cluster_size: int = 50,
    connectivity: int = 26,
    two_sided: bool = False,
    use_abs: bool = False,
    return_labels: bool = False,
):
    """
    Cluster thresholding for a 3D activation/stat map.

    Parameters
    ----------
    act : np.ndarray
        3D activation/stat map, shape (X, Y, Z).
    voxel_thresh : float
        Voxel-wise threshold. e.g. z>3.1; if your act is already a binary(0/1) map, set voxel_thresh=0.5.
    min_cluster_size : int
        Minimum number of voxels to keep a cluster (e.g. 50).
    connectivity : int
        6, 18, or 26. (3D neighborhood connectivity)
    two_sided : bool
        If True, threshold both positive and negative (|act| > voxel_thresh by default).
        If False, keep only positive side (act > voxel_thresh), unless use_abs=True.
    use_abs : bool
        If True, uses |act| > voxel_thresh regardless of two_sided. (common for magnitude maps)
    return_labels : bool
        If True, also return the labeled connected-components volume.

    Returns
    -------
    filtered_act : np.ndarray
        Activation map after removing small clusters. Same shape as act.
        Values are preserved for kept clusters; removed clusters are set to 0.
    info : dict
        Some statistics about clusters (kept/removed sizes, counts).
    labels (optional) : np.ndarray
        Labeled connected-components volume. 0 is background.
    """
    act = np.asarray(act)
    if act.ndim != 3:
        raise ValueError(f"`act` must be 3D, got shape {act.shape}")

    # Lazy import to keep dependency local
    from scipy import ndimage as ndi

    # 1) build supra-threshold mask
    finite = np.isfinite(act)
    if use_abs or two_sided:
        mask = finite & (np.abs(act) > voxel_thresh)
    else:
        mask = finite & (act > voxel_thresh)

    # 2) choose connectivity structure
    if connectivity == 6:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)  # faces
    elif connectivity == 26:
        structure = ndi.generate_binary_structure(rank=3, connectivity=2)  # faces+edges+corners
    elif connectivity == 18:
        # custom 18-neighborhood: faces + edges (no corners)
        structure = np.zeros((3, 3, 3), dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) <= 2:  # exclude (1,1,1) corners
                        structure[dx + 1, dy + 1, dz + 1] = True
    else:
        raise ValueError("`connectivity` must be one of {6, 18, 26}")

    # 3) label connected components
    labels, nlab = ndi.label(mask, structure=structure)

    if nlab == 0:
        filtered = np.zeros_like(act)
        info = dict(
            n_clusters=0,
            kept_clusters=0,
            removed_clusters=0,
            kept_voxels=0,
            removed_voxels=0,
            cluster_sizes=[],
            kept_sizes=[],
            removed_sizes=[],
        )
        return (filtered, info, labels) if return_labels else (filtered, info)

    # 4) compute cluster sizes
    counts = np.bincount(labels.ravel())  # counts[0] is background
    cluster_sizes = counts[1:]  # size for label 1..nlab

    # 5) decide which labels to keep
    keep = np.zeros(nlab + 1, dtype=bool)
    keep[0] = False
    keep[1:] = cluster_sizes >= min_cluster_size

    keep_mask = keep[labels]
    filtered = np.where(keep_mask, act, 0)

    kept_sizes = cluster_sizes[cluster_sizes >= min_cluster_size]
    removed_sizes = cluster_sizes[cluster_sizes < min_cluster_size]

    info = dict(
        n_clusters=int(nlab),
        kept_clusters=int(kept_sizes.size),
        removed_clusters=int(removed_sizes.size),
        kept_voxels=int(kept_sizes.sum()) if kept_sizes.size else 0,
        removed_voxels=int(removed_sizes.sum()) if removed_sizes.size else 0,
        cluster_sizes=cluster_sizes.tolist(),
        kept_sizes=kept_sizes.tolist(),
        removed_sizes=removed_sizes.tolist(),
    )

    return (filtered, info, labels) if return_labels else (filtered, info)

def cal_z_score(rec_zer,
                activation_handler, 
                dyn_datas,
                sim_conf, 
                threshold, 
                TR, 
                roi_resampled):
    from snake.toolkit.analysis.stats import contrast_zscore, get_scores
    waveform_name = f"activation-{activation_handler.event_name}"
    good_d = None
    for d in dyn_datas:
        print(d.name)
        if d.name == waveform_name:
            good_d = d
    if good_d is None:
        raise ValueError("No dynamic data found matching waveform name")

    bold_signal = good_d.data[0]
    bold_sample_time = np.arange(len(bold_signal)) * sim_conf.seq.TR / 1000
    TR_vol = TR
    z_score = contrast_zscore(rec_zer, TR_vol, bold_signal, bold_sample_time, activation_handler.event_name)
    stats_results = get_scores(z_score,roi_resampled, threshold)
    # np.save('z_score.npy', z_score)
    
    return z_score, stats_results

def plot_ts_roi(
    arr: np.ndarray,
    roi: np.ndarray,
    roi_idx: int,
    TR_s: float,
    ax=None,
    center: bool = False,
    label: str = None,
    **kwargs,
):
    """Plot time series of given ROI for one reconstruction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

    N = len(arr)
    time_samples = np.arange(N, dtype=np.float64) * TR_s
    arr_ = np.abs(arr)

    # Extract ROI signal
    arr_ = arr_[..., roi].squeeze()
    if roi_idx is None:
        roi_idx = np.arange(np.sum(roi))
    ts = arr_[:, roi_idx]

    # Compute TSNR
    if isinstance(roi_idx, np.ndarray):
        tsnr = np.median(np.mean(abs(ts), axis=0) / np.std(abs(ts), axis=0))
    else:
        tsnr = np.median(np.mean(abs(ts)) / np.std(abs(ts)))
    print(f"[{label}] TSNR:", tsnr)

    # Center / normalize
    if center:
        mean_val = np.mean(ts, axis=0)
        ts = (ts - mean_val) / np.std(abs(ts), axis=0)

    # Average across voxels
    if len(ts.shape) == 2:
        ts = np.mean(ts, axis=-1)

    marker_indices = np.linspace(0, N - 1, N, dtype=int)
    line, = ax.plot(
        time_samples, ts, marker='o', markersize=1.5, markevery=marker_indices,
        linewidth=1, label=label, **kwargs
    )
    return ax, ts, line

def get_f2_analysis(stats_dict: dict, target_threshold: float = 3.3) -> dict:
    """
    分析特定方法的 F2 分数，包括指定阈值分数和最大潜力分数。
    """
    tp = np.array(stats_dict["tp"])
    fp = np.array(stats_dict["fp"])
    fn = np.array(stats_dict["fn"])
    thresholds = np.array(stats_dict["tresh"])

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # 计算全量 F2 Scores
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-10)

    # 1. 计算固定阈值 3.0 的结果
    idx_fixed = np.abs(thresholds - target_threshold).argmin()
    f2_at_fixed = f2_scores[idx_fixed]
    actual_thresh_at_fixed = thresholds[idx_fixed] # 实际匹配到的最接近 3 的采样点

    # 2. 计算最大 F2 及其对应的阈值
    idx_max = np.argmax(f2_scores)
    f2_max = f2_scores[idx_max]
    best_threshold = thresholds[idx_max]

    return {
        "f2_fixed": f2_at_fixed,
        "actual_thresh": actual_thresh_at_fixed,
        "f2_max": f2_max,
        "best_thresh": best_threshold
    }

def get_f1_analysis(stats_dict: dict, target_threshold: float = 3.3) -> dict:
    """
    分析特定方法的 F1 分数，包括指定阈值分数和最大潜力分数。
    """
    tp = np.array(stats_dict["tp"])
    fp = np.array(stats_dict["fp"])
    fn = np.array(stats_dict["fn"])
    thresholds = np.array(stats_dict["tresh"])

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # 计算全量 F1 Scores (Dice Score)
    # 公式: 2 * (P * R) / (P + R)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)

    # 1. 计算固定阈值的结果
    idx_fixed = np.abs(thresholds - target_threshold).argmin()
    f1_at_fixed = f1_scores[idx_fixed]
    actual_thresh_at_fixed = thresholds[idx_fixed] 

    # 2. 计算最大 F1 及其对应的阈值
    idx_max = np.argmax(f1_scores)
    f1_max = f1_scores[idx_max]
    best_threshold = thresholds[idx_max]

    return {
        "f1_fixed": f1_at_fixed,
        "actual_thresh": actual_thresh_at_fixed,
        "f1_max": f1_max,
        "best_thresh": best_threshold
    }