"""Train / validation / test flight splits, including the held-out operating-condition splits.

    random      the paper's split (`data_utils.get_flights_and_ranges`, test list repeats 269 and 278)
    payload500  test = all 500 g flights            (extrapolation in payload)
    payload250  test = all 250 g flights            (interpolation in payload)
    speed12     test = all 12 m/s flights           (extrapolation in speed)
    alt100      test = all 100 m flights            (extrapolation in altitude; the 2 mixed 25-50-100-25 m
                                                     flights are dropped, since they also fly at 100 m)
    route       test = the 10 flights off route 5   (the paper's "random" flights: long route + test route)

For the held-out splits the validation set is drawn (seed 42, 25 %) from the remaining flights, so it stays
inside the training conditions, and every flight appears at most once. Flight 232 is never a held-out test
flight: its last ~80 logged steps are after the motors stop (power <= 0 W), which makes its MAPE infinite.

    python splits.py            # prints the size and conditions of every split
"""
import os

import numpy as np
import pandas as pd

from data_utils import get_flights_and_ranges

SPLITS = ['random', 'payload500', 'payload250', 'speed12', 'alt100', 'route']
FLIGHT_SHEET = 'Flight Sheet.xlsx'
NO_MAPE_FLIGHTS = [232]  # power <= 0 W at some steps; kept for training, never a held-out test flight


def load_conditions(data_dir='./data/'):
    """Commanded payload (g), speed (m/s), altitude (m, str: '25-50-100-25' for mixed) and route per flight."""
    sheet = pd.read_excel(os.path.join(data_dir, FLIGHT_SHEET))
    sheet = sheet.dropna(subset=['Flight #'])
    return pd.DataFrame({
        'payload': sheet['Payload [g]'].values,
        'speed': sheet['Speed [m/s]'].values,
        'altitude': sheet['Altitude [m]'].astype(str).values,
        'route': sheet['Route #'].astype(str).values,
    }, index=sheet['Flight #'].astype(int).values)


def _held_out(cond, split):
    """(test mask, excluded mask) over `cond`'s rows."""
    none = np.zeros(len(cond), dtype=bool)
    if split == 'payload500':
        return cond['payload'] == 500, none
    if split == 'payload250':
        return cond['payload'] == 250, none
    if split == 'speed12':
        return cond['speed'] == 12, none
    if split == 'alt100':
        return cond['altitude'] == '100', cond['altitude'].str.contains('-')
    if split == 'route':
        return cond['route'] != '5', none
    raise ValueError('unknown split {!r}, expected one of {}'.format(split, SPLITS))


def get_split(all_data, split='random', data_dir='./data/', val_split=0.25, random_seed=42):
    """(test_range, val_range, train_range) arrays of flight numbers for `split`."""
    if split == 'random':
        _, _, _, test_range, val_range, train_range = get_flights_and_ranges(all_data, val_split=val_split)
        return test_range, val_range, train_range

    cond = load_conditions(data_dir).loc[sorted(all_data)]
    test, excluded = _held_out(cond, split)
    unscorable = cond.index.isin(NO_MAPE_FLIGHTS)
    excluded, test = excluded | (test & unscorable), test & ~unscorable
    test_range = cond.index[test].values
    pool = cond.index[~test & ~excluded].values

    rng = np.random.RandomState(random_seed)
    val_range = np.sort(rng.choice(pool, size=int(np.ceil(len(pool) * val_split)), replace=False))
    train_range = np.setdiff1d(pool, val_range)
    return test_range, val_range, train_range


if __name__ == '__main__':
    from data_utils import load_dataset
    all_data = load_dataset('./data/')
    cond = load_conditions('./data/').loc[sorted(all_data)]
    rows = []
    for split in SPLITS:
        test, val, train = get_split(all_data, split)
        t = cond.loc[np.unique(test)]
        rows.append({'split': split, 'train': len(train), 'val': len(val), 'test': len(np.unique(test)),
                     'test payloads (g)': sorted(t['payload'].unique().astype(int)),
                     'test speeds (m/s)': sorted(t['speed'].unique()),
                     'test altitudes (m)': sorted(t['altitude'].unique())})
    print(pd.DataFrame(rows).to_string(index=False))
