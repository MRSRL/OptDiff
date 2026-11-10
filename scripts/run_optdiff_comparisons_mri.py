"""Run the MRI cells of the paper's comparison tables and the proposed-schedule column of its noise-schedule table, and
summarize them.

Every method entry of configs/mri/tuned/fastmri.yaml (acceleration, NFE, schedule, method) is one table cell; the
entries grouped under optimized_schedule run the baselines with OptDiff's noise schedule. A cell runs
main_mri.py once per test volume, with the cell's tuned values from the table (utils/task_config.py); --dry-run prints
the full commands. Each volume writes to its own folder, so an interrupted run resumes with the volumes that have no
complete metrics.txt. summary.md in the output directory lists the mean PSNR (magnitude) / SSIM of every table cell
run so far, over all slices.

    python scripts/run_optdiff_comparisons_mri.py --devices 0 1 2        # every cell, one queue per GPU
    python scripts/run_optdiff_comparisons_mri.py --accels 8 --steps 100 --methods optdiff dps --dry-run
    python scripts/run_optdiff_comparisons_mri.py --schedules optimized   # only the noise-schedule table
    python scripts/run_optdiff_comparisons_mri.py --summary-only

Run from the repository root.
"""
import argparse
import os
import queue
import re
import shlex
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from utils.task_config import load_table, tuned_overrides

DATASET = 'fastmri'
ACCELS = [4, 8]
METHODS = ['optdiff', 'reddiff', 'dps', 'pigdm', 'daps', 'diffpir']
METHOD_NAMES = {'optdiff': 'OptDiff', 'reddiff': 'RED-diff', 'dps': 'DPS', 'pigdm': 'PiGDM', 'daps': 'DAPS',
                'diffpir': 'DiffPIR'}
SCHEDULES = {'standard': None, 'optimized': 'optimized_schedule'}  # schedule name: variant group in the table
NUM_SLICES = 11  # slices per test volume (data_config.slice of configs/mri/fastmri.yaml)


def table_cells():
    """The (acceleration, NFE, schedule, method) of every method entry in the tuned table."""
    table = load_table('mri', DATASET)
    cells = []
    for rkey, entry in table.items():
        for skey, methods in entry.items():
            if not skey.startswith('steps'):
                continue
            for schedule, variant in SCHEDULES.items():
                group = methods if variant is None else methods.get(variant, {})
                cells += [(int(rkey[1:]), int(skey[len('steps'):]), schedule, method)
                          for method in METHODS if method in group]
    return cells


def cell_dir(root, accel, steps, schedule, method):
    sub = [] if schedule == 'standard' else [f'{schedule}_schedule']
    return os.path.join(root, f'R_{accel}', f'steps_{steps}', *sub, method)


def volume_dir(root, accel, steps, schedule, method, volume):
    return os.path.join(cell_dir(root, accel, steps, schedule, method), f'volume_{volume:02d}')


def read_rows(out):
    """(psnr, psnr_abs, ssim) of every slice in a volume folder's metrics.txt."""
    path = os.path.join(out, 'metrics.txt')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()
    return [tuple(float(v) for v in m) for m in
            re.findall(r'^\(\d+,.*\( *([-\d.]+), *([-\d.]+), *([-\d.]+) *\)', text, flags=re.M)]


def command(accel, steps, schedule, method, volume, device, out, args):
    return [sys.executable, 'main_mri.py', '--config-name', DATASET, f'algo={method}',
            f'noise_schedule_config.num_steps={steps}',
            *tuned_overrides('mri', DATASET, f'R{accel}', steps, method, SCHEDULES[schedule]),
            f'data_config.file={volume}', f'model_config.device={device}',
            f'output_config.save_images={args.save_images}', f'output_config.output_dir={out}',
            f'hydra.run.dir={out}/hydra']


def run_jobs(jobs, args):
    """Run the jobs (one volume of one cell each), one queue per device."""
    todo = queue.Queue()
    for job in jobs:
        todo.put(job)
    lock = threading.Lock()

    def worker(device):
        while True:
            try:
                accel, steps, schedule, method, volume = todo.get_nowait()
            except queue.Empty:
                return
            out = volume_dir(args.output_dir, accel, steps, schedule, method, volume)
            os.makedirs(out, exist_ok=True)
            # main_mri appends to metrics.txt and runtimes.txt: set aside those of an unfinished earlier run
            for name in ('metrics.txt', 'runtimes.txt'):
                path = os.path.join(out, name)
                if os.path.exists(path):
                    os.replace(path, path + '.interrupted')
            label = f'R={accel} {steps} NFE {schedule} schedule {method} volume {volume}'
            with open(os.path.join(out, 'log.txt'), 'w') as log:
                code = subprocess.run(command(accel, steps, schedule, method, volume, device, out, args),
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            with lock:
                status = 'done' if code == 0 else f'FAILED (exit code {code}, see {out}/log.txt)'
                print(f'[device {device}] {status}: {label}', flush=True)

    threads = [threading.Thread(target=worker, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def cell_result(cells, args, accel, steps, schedule, method):
    """Mean +- std of PSNR (magnitude) / SSIM of a cell over the slices run so far, or '-' / 'not run'."""
    if (accel, steps, schedule, method) not in cells:
        return '-'
    rows = [r for v in range(args.volumes)
            for r in read_rows(volume_dir(args.output_dir, accel, steps, schedule, method, v))]
    if not rows:
        return 'not run'
    vals = np.array(rows)
    cell = f'{vals[:, 1].mean():.2f} +- {vals[:, 1].std():.2f} / {vals[:, 2].mean():.3f} +- {vals[:, 2].std():.3f}'
    if len(rows) < args.volumes * NUM_SLICES:
        cell += f' (n={len(rows)})'
    return cell


def summary(cells, args):
    """Markdown tables of all table cells, finished or not: per NFE, methods x accelerations; then the noise-schedule
    table, methods x schedules."""
    lines = ['# MRI table cells', '',
             'Mean +- std of PSNR (magnitude) / SSIM over the slices of all test volumes run so far. `-`: no tuned values '
             'in configs/mri/tuned; `not run`: no volume finished yet.',
             f'`(n=..)`: fewer than {args.volumes * NUM_SLICES} slices.', '']
    for steps in sorted({s for _, s, schedule, _ in cells if schedule == 'standard'}):
        lines += [f'## NFE {steps}', '', '| Method | ' + ' | '.join(f'R={r}' for r in ACCELS) + ' |',
                  '| --- |' + ' --- |' * len(ACCELS)]
        for method in METHODS:
            row = [cell_result(cells, args, accel, steps, 'standard', method) for accel in ACCELS]
            lines.append(f'| {METHOD_NAMES[method]} | ' + ' | '.join(row) + ' |')
        lines.append('')
    for accel, steps in sorted({(a, s) for a, s, schedule, _ in cells if schedule == 'optimized'}):
        lines += [f'## Noise schedules, R={accel}, NFE {steps}', '',
                  '| Method | Standard | Optimized |', '| --- | --- | --- |']
        for method in METHODS:
            standard = cell_result(cells, args, accel, steps, 'standard', method)
            # OptDiff runs with its optimized schedule in the main table
            optimized = standard if method == 'optdiff' else cell_result(cells, args, accel, steps, 'optimized', method)
            lines.append(f'| {METHOD_NAMES[method]} | ' + ('-' if method == 'optdiff' else standard) + f' | {optimized} |')
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accels', nargs='+', type=int, default=ACCELS, choices=ACCELS)
    parser.add_argument('--steps', nargs='+', type=int, default=[20, 100], help='NFE')
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--schedules', nargs='+', default=list(SCHEDULES), choices=list(SCHEDULES),
                        help='standard: the comparison tables; optimized: the baselines of the noise-schedule table')
    parser.add_argument('--devices', nargs='+', type=int, default=[0], help='GPUs; one run at a time on each')
    parser.add_argument('--volumes', type=int, default=30, help='test volumes 0 .. N-1 (the paper uses 30)')
    parser.add_argument('--save-images', action='store_true', help='also save the reconstructions as PNG')
    parser.add_argument('--output-dir', default=os.path.abspath('outputs/optdiff_comparisons/mri'))
    parser.add_argument('--dry-run', action='store_true', help='print the commands of the volumes to run')
    parser.add_argument('--summary-only', action='store_true', help='only write summary.md')
    args = parser.parse_args()

    if not os.path.exists('main_mri.py'):
        sys.exit('Run from the repository root.')

    cells = set(table_cells())
    jobs = [(accel, steps, schedule, method, volume) for accel, steps, schedule, method in sorted(cells)
            if accel in args.accels and steps in args.steps and schedule in args.schedules and method in args.methods
            for volume in range(args.volumes)
            if len(read_rows(volume_dir(args.output_dir, accel, steps, schedule, method, volume))) < NUM_SLICES]
    jobs.sort(key=lambda job: -job[1])  # longest runs first

    if args.dry_run:
        for accel, steps, schedule, method, volume in jobs:
            out = volume_dir(args.output_dir, accel, steps, schedule, method, volume)
            print(shlex.join(command(accel, steps, schedule, method, volume, args.devices[0], out, args)))
        print(f'{len(jobs)} volumes to run')
        return
    if not args.summary_only:
        print(f'{len(jobs)} volumes to run on devices {args.devices}')
        run_jobs(jobs, args)

    text = summary(cells, args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'summary.md'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
