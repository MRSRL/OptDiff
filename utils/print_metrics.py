"""Print the mean +- std of the metrics in metrics.txt files of main_mri.py (PSNR, magnitude PSNR, SSIM) or
main_natimg.py (PSNR, SSIM, LPIPS), and of the reconstruction time if logged. The column names come from the file's
title line. Rows of several settings in one file (e.g. a Hydra multirun into one folder) are summarized per setting,
labelled by the values that differ; an image (its DATA values: MRI file and slice, natural-image slice) logged more
than once with the same settings counts once, with its last row.

Usage:
    python utils/print_metrics.py <metrics_file> [<metrics_file2> ...]
"""
import re
import sys
import numpy as np

NOT_SETTINGS = ('DATA', 'METRICS', 'TIME')


def split_sections(line):
    """The comma-separated parts of a metrics.txt line that lie outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(line):
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            parts.append(line[start:i].strip())
            start = i + 1
    parts.append(line[start:].strip())
    return [p for p in parts if p]


def number(text):
    return float(re.sub(r'^(-?)0*(inf|nan)$', r'\1\2', text))  # zero-padded formats write -inf as -00inf


def read(path):
    """Metric names, setting names and rows ({settings: {DATA values: metric values}}) of a metrics.txt."""
    sections, rows = None, {}
    with open(path) as f:
        for line in f:
            if 'METRICS (' in line and not line.startswith('('):  # title: SECTION (field, ...) or SECTION (one field)
                sections = []
                for part in split_sections(line):
                    m = re.fullmatch(r'(.*?)\s*\((.*)\)', part)
                    sections.append((m.group(1), [s.strip() for s in m.group(2).split(',')]) if m else (part, [part]))
            elif line.startswith('(') and sections:
                values = [[v.strip() for v in part.strip('() ').split(',')] for part in split_sections(line)]
                if len(values) != len(sections):
                    continue
                settings, metrics, key = [], [], None
                for (name, _), vals in zip(sections, values):
                    if name == 'DATA':
                        key = tuple(vals[:2])
                    elif name in ('METRICS', 'TIME'):
                        metrics += [number(v) for v in vals]
                    else:
                        settings += vals
                group = rows.setdefault(tuple(settings), {})
                group[key if key is not None else len(group)] = metrics
    if sections is None:
        return [], [], {}
    metric_names = [field for name, fields in sections if name == 'METRICS' for field in fields]
    metric_names += ['time (s)' for name, _ in sections if name == 'TIME']
    setting_names = [field for name, fields in sections if name not in NOT_SETTINGS for field in fields]
    return metric_names, setting_names, rows


def summarize(path):
    metric_names, setting_names, rows = read(path)
    if not rows:
        print(f"{path}: no metric lines found")
        return
    print(path)
    varying = [i for i in range(len(setting_names)) if len({settings[i] for settings in rows}) > 1]
    for settings, by_slice in rows.items():
        values = np.array(list(by_slice.values()))
        label = ', '.join(f'{setting_names[i]}={settings[i]}' for i in varying)
        print(f"  {label + ': ' if label else ''}{len(values)} rows")
        with np.errstate(invalid='ignore'):  # diverged runs log -inf / nan
            for name, column in zip(metric_names, values.T):
                print(f"    {name:9s} {column.mean():.4f} +- {column.std():.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python utils/print_metrics.py <metrics_file> [<metrics_file2> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        summarize(path)
