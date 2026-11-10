"""Config helpers: the check of required values used by main_natimg.py and main_mri.py, and the tuned values of the
paper's table cells (configs/<domain>/tuned/<dataset>.yaml) as command-line overrides, for the drivers in scripts/.

A table holds, per task (natural images: data_config.deg; MRI: R<acceleration>), the config values shared by its
cells (e.g. data_config) and, per number of steps (steps<N>), one block per method, optionally grouped under a variant
name (e.g. MRI optimized_schedule). A method block maps config sections to their values; a block of plain values
(the natural-image baselines) is short for baseline_config. A driver passes these overrides before its own, so the
values it adds after them take precedence.
"""
import os

from omegaconf import OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


def missing_keys(config):
    """Keys whose value is ??? (OmegaConf.missing_keys resolves interpolations, and fails on one that points to a
    ??? value, e.g. optimizer_config.num_steps when noise_schedule_config.num_steps is missing)."""
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f'{path}{key}.')
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from walk(value, f'{path}{i}.')
        elif node == '???':
            yield path[:-1]
    return sorted(walk(OmegaConf.to_container(config, resolve=False), ''))


def load_table(domain, dataset):
    """The tuned table of a dataset ('natimg' / 'mri' domain) as nested dicts and lists."""
    return OmegaConf.to_container(OmegaConf.load(os.path.join(CONFIG_DIR, domain, 'tuned', f'{dataset}.yaml')))


def format_value(value):
    """A table value in Hydra's override grammar, parsing back to the same value and type (strings are written as
    they are: the tables hold plain names)."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, float):
        return repr(value)  # shortest string that parses back to the same float
    if isinstance(value, list):
        return '[' + ','.join(format_value(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ','.join(f'{k}:{format_value(v)}' for k, v in value.items()) + '}'
    return str(value)


def section_overrides(section, values):
    """Overrides setting the values of one config section, one per leaf."""
    if section == 'optimizer_weights_config':
        # One value: the weights replace the null of the OptDiff algo config
        return [f'{section}={format_value(values)}']
    overrides = []
    for key, value in values.items():
        if isinstance(value, dict):
            overrides += section_overrides(f'{section}.{key}', value)
        else:
            overrides.append(f'{section}.{key}={format_value(value)}')
    return overrides


def tuned_overrides(domain, dataset, task, num_steps, algo, variant=None):
    """main_natimg.py / main_mri.py overrides with the tuned values of one table cell: the values shared by the
    task's cells and the settings of the method (in the variant group, if given)."""
    table = load_table(domain, dataset)
    entry = table.get(task, {}).get(f'steps{num_steps}') or {}
    where = f'{task}.steps{num_steps}'
    if variant is not None:
        entry = entry.get(variant) or {}
        where += f'.{variant}'
    if algo not in entry:
        raise KeyError(f'configs/{domain}/tuned/{dataset}.yaml has no {algo} entry for {where}. Add one, or run the '
                       f'main with the values on the command line (it lists the ones it needs).')
    overrides = []
    for section, values in table[task].items():
        if not section.startswith('steps'):
            overrides += section_overrides(section, values)
    block = entry[algo]
    if not all(isinstance(v, dict) for v in block.values()):
        block = {'baseline_config': block}
    for section, values in block.items():
        overrides += section_overrides(section, values)
    return overrides
