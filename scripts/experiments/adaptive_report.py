"""Write an interpretation of a completed paired comparison beside its numeric results."""
import argparse
from pathlib import Path

import pandas as pd


def report(folder):
    data = pd.read_csv(folder/'per_flight.csv')
    flight = data[data.phase == 'flight']
    target = 'adaptive16-leaky' if 'adaptive16-leaky' in set(data.arch) else 'residual16-leaky'
    metrics = ['mape','regime_re','signed_wh','absolute_wh','underprediction_wh']
    sections = ['# ' + folder.name.replace('_',' ').capitalize() + ' comparison', '']
    for (split, size), group in flight.groupby(['split','train_size']):
        sections.extend([f'## {split}, {size} training flights', '',
                         f'{group.flight.nunique()} distinct scored flights. The omitted warmup averages '
                         f'{group.omitted_s.mean():.2f} seconds. Values average equally over flights within '
                         'each run, then equally over runs.', ''])
        runs = group.groupby(['arch','seed','subset_seed'])[metrics].mean()
        sections.extend([runs.groupby('arch').mean().to_markdown(floatfmt='.3f'), ''])
        for base in sorted(set(group.arch)-{target}):
            keyed = group.set_index(['arch','seed','subset_seed','flight'])
            diff = keyed.loc[target,metrics] - keyed.loc[base,metrics]
            if diff.isna().any().any():
                raise ValueError('incomplete paired comparison')
            means = diff.groupby(['seed','subset_seed']).mean()
            d = means.mean()
            sections.extend([f'Against {base}, {target} changed mean absolute energy error by '
                             f'{d.absolute_wh:+.3f} Wh and mean underprediction magnitude by '
                             f'{d.underprediction_wh:+.3f} Wh. Absolute error fell in '
                             f'{(means.absolute_wh < 0).sum()} of {len(means)} runs and '
                             f'{(diff.absolute_wh < 0).sum()} of {len(diff)} matched flight/run observations.', ''])
        phases = data[(data.split == split) & (data.train_size == size) & (data.phase != 'flight')]
        columns = ['bias_w','signed_wh']
        if 'induced_mult' in phases:
            columns += [term+'_correction_wh' for term in ('induced','profile','parasitic')]
        means = phases.groupby(['arch','phase','seed','subset_seed'])[columns].mean().groupby(['arch','phase']).mean()
        sections.extend([means.to_markdown(floatfmt='.3f'), ''])
        for base in sorted(set(group.arch)-{target}):
            difference = means.loc[target,'signed_wh'] - means.loc[base,'signed_wh']
            phase = difference.abs().idxmax()
            sections.extend([f'{phase.capitalize()} contributes the largest absolute change in signed '
                             f'energy relative to {base} ({difference[phase]:+.3f} Wh per flight).', ''])
    sections.append('Signed error is predicted minus measured. Underprediction magnitude clips each '
                    'scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed '
                    'error. Only total power is measured, so term corrections describe the fitted model.')
    (folder/'interpretation.md').write_text('\n'.join(sections)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    report(parser.parse_args().folder)
