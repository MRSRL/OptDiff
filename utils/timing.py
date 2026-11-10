"""Runtimes of the reconstructions, measured the same way by main_mri.py and main_natimg.py.

Only the step loop is timed (not the initial estimate). Every runtime is appended to runtimes.txt in the output
directory, one line per reconstruction. The mean and std leave out the first reconstruction of the run, which also
pays for one-time GPU setup.
"""
import os
import time
import numpy as np
import torch


def clock(device):
    """Wall-clock time in seconds, read after the device has finished its queued work."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def save_runtime(output_dir, runtime):
    with open(os.path.join(output_dir, 'runtimes.txt'), 'a') as f:
        f.write(f"{runtime}\n")


def runtime_summary(runtimes):
    """Mean and std of the runtimes without the first one."""
    if len(runtimes) < 2:
        return 'Average time: n/a (needs two or more reconstructions; the first one includes GPU warm-up)'
    rest = np.array(runtimes[1:])
    return f'Average time: {rest.mean():.4f} +- {rest.std():.4f} (s), without the first reconstruction (GPU warm-up)'
