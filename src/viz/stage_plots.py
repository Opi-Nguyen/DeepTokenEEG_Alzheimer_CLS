import numpy as np
import mne
from src.utils.plotting import plot_timeseries_compare, plot_heatmap

def welch_psd_1d(x, fs, n_fft=512):
    psd, freqs = mne.time_frequency.psd_array_welch(
        x.astype(np.float64), sfreq=fs, n_fft=n_fft, verbose="ERROR"
    )
    return freqs, psd

def plot_raw_vs_filtered_vs_band(raw_1d, filtered_1d, band_1d, out_dir, tag):
    plot_timeseries_compare(
        [raw_1d, filtered_1d, band_1d],
        labels=["raw", "broad_filtered", "band_extracted"],
        title=f"Time series compare ({tag})",
        out_path=f"{out_dir}/{tag}_timeseries.png",
        max_points=2000
    )

def plot_psd_compare(raw_1d, filtered_1d, band_1d, fs, out_dir, tag):
    fr, ps_raw = welch_psd_1d(raw_1d, fs)
    ff, ps_flt = welch_psd_1d(filtered_1d, fs)
    fb, ps_band = welch_psd_1d(band_1d, fs)

    # reuse line plot
    import matplotlib.pyplot as plt
    from src.utils.plotting import savefig
    plt.figure()
    plt.semilogy(fr, ps_raw, label="raw")
    plt.semilogy(ff, ps_flt, label="broad_filtered")
    plt.semilogy(fb, ps_band, label="band_extracted")
    plt.legend()
    plt.title(f"PSD compare ({tag})")
    plt.xlabel("Hz")
    plt.ylabel("PSD")
    savefig(f"{out_dir}/{tag}_psd.png")

def plot_segment_heatmap(seg_tc, out_dir, tag):
    # seg_tc: [T, C] -> visualize C x T
    mat = seg_tc.T
    plot_heatmap(mat, title=f"Segment heatmap ({tag})", out_path=f"{out_dir}/{tag}_segment_heatmap.png",
                 xlabel="Time", ylabel="Channel")
