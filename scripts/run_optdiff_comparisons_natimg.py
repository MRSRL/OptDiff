"""Run the natural-image cells of the paper's comparison tables and summarize them.

Every method entry of configs/natimg/tuned/{imagenet,ffhq}.yaml (dataset, task, number of steps, method) is one table
cell. A cell runs main_natimg.py with those choices and the cell's tuned values from the table
(utils/task_config.py); --dry-run prints the full commands.

Cells whose output already has final metrics are skipped, so an interrupted run resumes where it stopped.
summary.md in the output directory lists the mean PSNR / SSIM / LPIPS of every table cell run so far.

    python scripts/run_optdiff_comparisons_natimg.py --devices 0 1 2        # every cell, one queue per GPU
    python scripts/run_optdiff_comparisons_natimg.py --datasets ffhq --steps 20 --methods optdiff --dry-run
    python scripts/run_optdiff_comparisons_natimg.py --summary-only

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from utils.task_config import load_table, tuned_overrides

DATASETS = ['imagenet', 'ffhq']
METHODS = ['optdiff', 'ddnm', 'diffpir', 'dps', 'ddrm', 'reddiff']
TASKS = ['sr4', 'sr16', 'deblur_gauss_window6_sigma3', 'deblur_gauss_window12_sigma3']
TASK_NAMES = {'sr4': 'SR x4', 'sr16': 'SR x16', 'deblur_gauss_window6_sigma3': 'Deblur (6, 3)',
              'deblur_gauss_window12_sigma3': 'Deblur (12, 3)'}
METHOD_NAMES = {'optdiff': 'OptDiff', 'ddnm': 'DDNM', 'diffpir': 'DiffPIR', 'dps': 'DPS', 'ddrm': 'DDRM',
                'reddiff': 'RED-diff'}


def table_cells(dset):
    """The (task, steps, method) of every method entry in the dataset's tuned table."""
    table = load_table('natimg', dset)
    return [(deg, int(skey[len('steps'):]), method)
            for deg, entry in table.items() for skey, methods in entry.items() if skey.startswith('steps')
            for method in METHODS if method in methods]


def cell_dir(root, dset, deg, steps, method):
    return os.path.join(root, dset, f'steps_{steps}', deg, method)


def read_metrics(out):
    """(means and stds of PSNR / SSIM / LPIPS, number of images) of a finished cell, or None."""
    path = os.path.join(out, 'metrics.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    values = {}
    for name in ('PSNR', 'SSIM', 'LPIPS'):
        m = re.search(rf'Average {name}: (\S+) \+- (\S+)', text)
        if m is None:
            return None
        values[name] = (float(m.group(1)), float(m.group(2)))
    num_images = len(re.findall(r'^\(\d+\b', text, flags=re.M))  # rows start with (slice; older runs (file, slice, ...
    return values, num_images


def command(dset, deg, steps, method, device, out, args):
    return [sys.executable, 'main_natimg.py', '--config-name', dset, f'algo={method}', f'data_config.deg={deg}',
            f'noise_schedule_config.num_steps={steps}', *tuned_overrides('natimg', dset, deg, steps, method),
            f'data_config.slice={args.num_images}', f'model_config.device={device}',
            f'output_config.save_images={args.save_images}', f'output_config.output_dir={out}',
            f'hydra.run.dir={out}/hydra']


def run_cells(jobs, args):
    """Run the jobs, one queue per device."""
    todo = queue.Queue()
    for job in jobs:
        todo.put(job)
    lock = threading.Lock()

    def worker(device):
        while True:
            try:
                dset, deg, steps, method = todo.get_nowait()
            except queue.Empty:
                return
            out = cell_dir(args.output_dir, dset, deg, steps, method)
            os.makedirs(out, exist_ok=True)
            # main_natimg appends to metrics.txt and runtimes.txt: set aside those of an unfinished earlier run
            for name in ('metrics.txt', 'runtimes.txt'):
                path = os.path.join(out, name)
                if os.path.exists(path):
                    os.replace(path, path + '.interrupted')
            label = f'{dset} {deg} {steps} steps {method}'
            with lock:
                print(f'[device {device}] start {label}', flush=True)
            with open(os.path.join(out, 'log.txt'), 'w') as log:
                code = subprocess.run(command(dset, deg, steps, method, device, out, args),
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            with lock:
                status = 'done' if code == 0 else f'FAILED (exit code {code}, see {out}/log.txt)'
                print(f'[device {device}] {status}: {label}', flush=True)

    threads = [threading.Thread(target=worker, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def summary(cells, args):
    """Markdown tables of all table cells, finished or not: per dataset and number of steps, methods x tasks."""
    lines = ['# Natural-image table cells', '',
             'Mean PSNR / SSIM / LPIPS per cell. `-`: no tuned values in configs/natimg/tuned; `not run`: no final metrics yet.',
             f'`(n=..)`: fewer than {args.num_images} images.', '']
    for dset in DATASETS:
        dset_cells = cells[dset]
        for steps in sorted({s for _, s, _ in dset_cells}):
            tasks = [deg for deg in TASKS if any(c[0] == deg and c[1] == steps for c in dset_cells)]
            lines += [f'## {dset}, {steps} steps', '',
                      '| Method | ' + ' | '.join(TASK_NAMES[deg] for deg in tasks) + ' |',
                      '| --- |' + ' --- |' * len(tasks)]
            for method in METHODS:
                row = []
                for deg in tasks:
                    if (deg, steps, method) not in dset_cells:
                        row.append('-')
                        continue
                    result = read_metrics(cell_dir(args.output_dir, dset, deg, steps, method))
                    if result is None:
                        row.append('not run')
                        continue
                    values, num_images = result
                    cell = ' / '.join(f'{values[name][0]:.{digits}f}' for name, digits in
                                      (('PSNR', 2), ('SSIM', 3), ('LPIPS', 3)))
                    if num_images < args.num_images:
                        cell += f' (n={num_images})'
                    row.append(cell)
                lines.append(f'| {METHOD_NAMES[method]} | ' + ' | '.join(row) + ' |')
            lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--steps', nargs='+', type=int, default=[20, 100])
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--devices', nargs='+', type=int, default=[0], help='GPUs; one run at a time on each')
    parser.add_argument('--num-images', type=int, default=330, help='images 0 .. N-1 (the paper uses 330)')
    parser.add_argument('--save-images', action='store_true', help='also save the reconstructions as PNG')
    parser.add_argument('--output-dir', default=os.path.abspath('outputs/optdiff_comparisons/natimg'))
    parser.add_argument('--dry-run', action='store_true', help='print the commands of the cells to run')
    parser.add_argument('--summary-only', action='store_true', help='only write summary.md')
    args = parser.parse_args()

    if not os.path.exists('main_natimg.py'):
        sys.exit('Run from the repository root.')

    cells = {dset: table_cells(dset) for dset in DATASETS}
    jobs = [(dset, deg, steps, method) for dset in args.datasets for deg, steps, method in cells[dset]
            if deg in args.tasks and steps in args.steps and method in args.methods
            and read_metrics(cell_dir(args.output_dir, dset, deg, steps, method)) is None]
    jobs.sort(key=lambda job: -job[2])  # longest runs first

    if args.dry_run:
        for dset, deg, steps, method in jobs:
            out = cell_dir(args.output_dir, dset, deg, steps, method)
            print(shlex.join(command(dset, deg, steps, method, args.devices[0], out, args)))
        print(f'{len(jobs)} cells to run')
        return
    if not args.summary_only:
        print(f'{len(jobs)} cells to run on devices {args.devices}')
        run_cells(jobs, args)

    text = summary(cells, args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'summary.md'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
