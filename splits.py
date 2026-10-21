"""Train / validation / test flight splits, including the held-out operating-condition splits.

    random      the paper's split (`data_utils.get_flights_and_ranges`, test list repeats 269 and 278)
    payload500  test = all 500 g flights            (extrapolation in payload)
    payload250  test = all 250 g flights            (interpolation in payload)
    speed12     test = all 12 m/s flights           (extrapolation in speed)
    alt100      test = all 100 m flights            (extrapolation in altitude; the 2 mixed 25-50-100-25 m
                                                     flights are dropped, since they also fly at 100 m)
    route       test = all flights off the main route
    speed10-12  test = all 10 and 12 m/s flights    (wider speed extrapolation: train on 4-8 m/s)
    speed8-12   test = all 8, 10 and 12 m/s flights (train on 4-6 m/s only)
    random_subsampled, random_sub_<split>  data-size controls: random split, train cut to that split's size

For the held-out splits the validation set is drawn (seed 42, 25 %) from the remaining flights, so it stays
inside the training conditions, and every flight appears at most once. The condition splits (payload, speed,
altitude) use main-route flights only, so off-route flights appear in `route` and nowhere else. Flight 232 is
never a held-out test flight: its last ~80 logged steps are after the motors stop (power <= 0 W), which makes
its MAPE infinite.

Routes come from KiltHub's parameters.csv (Rodrigues et al. 2021) when it is available, since the flight sheet
mislabels some main-route flights (e.g. 276, 277 as "Test"). It is read from data/kilthub/parameters.csv
(scripts/download_data.sh puts it there) or downloaded once; if neither works, the flight sheet is used and a
warning is printed. The main route is the most common route label in whichever source is used.

    python splits.py            # prints the route source and the size and conditions of every split
"""
import hashlib
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd

from data_utils import get_flights_and_ranges

SPLITS = ['random', 'payload500', 'payload250', 'speed12', 'alt100', 'route',
          'speed10-12', 'speed8-12', 'random_subsampled', 'random_sub_speed10-12', 'random_sub_speed8-12']
CONTROLS = {'random_subsampled': 'payload500',
            'random_sub_speed10-12': 'speed10-12',
            'random_sub_speed8-12': 'speed8-12'}

FLIGHT_SHEET = 'Flight Sheet.xlsx'
NO_MAPE_FLIGHTS = [232]  # power <= 0 W at some steps; kept for training, never a held-out test flight

KILTHUB_PARAMS = os.path.join('kilthub', 'parameters.csv')
KILTHUB_MD5 = '7daa897a1af7847b592ccb7a459477ee'  # from api.figshare.com/v2/articles/12683453
KILTHUB_URLS = ['https://ndownloader.figshare.com/files/26385034',
                'https://kilthub.cmu.edu/ndownloader/files/26385034']


def _md5(raw):
    return hashlib.md5(raw).hexdigest()


def fetch_kilthub_params(data_dir='./data/', download=True):
    """Path to a checksum-verified KiltHub parameters.csv, downloading it if needed; None if unavailable."""
    path = os.path.join(data_dir, KILTHUB_PARAMS)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            if _md5(f.read()) == KILTHUB_MD5:
                return path
        warnings.warn('{} does not match its published checksum; ignoring it'.format(path))
    if not download:
        return None
    for url in KILTHUB_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                raw = r.read()
        except Exception as e:  # network down, server error, ...
            last_error = e
            continue
        if _md5(raw) != KILTHUB_MD5:
            last_error = 'checksum mismatch ({} bytes)'.format(len(raw))
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '.tmp', 'wb') as f:
            f.write(raw)
        os.replace(path + '.tmp', path)
        return path
    warnings.warn('could not download KiltHub parameters.csv ({})'.format(last_error))
    return None


def _kilthub_routes(path):
    """{flight: route label} from parameters.csv, or None if its columns can't be identified."""
    params = pd.read_csv(path)
    cols = {c.lower(): c for c in params.columns}
    flight_col = next((cols[c] for c in cols if 'flight' in c), None)
    route_col = next((cols[c] for c in cols if 'route' in c), None)
    if flight_col is None or route_col is None:
        warnings.warn('no flight/route columns in {} (columns: {})'.format(path, list(params.columns)))
        return None
    params = params.dropna(subset=[flight_col, route_col])
    return dict(zip(params[flight_col].astype(int), params[route_col].astype(str).str.strip()))


def load_conditions(data_dir='./data/', download=True):
    """Commanded payload (g), speed (m/s), altitude (m, str: '25-50-100-25' for mixed), route and
    main_route (bool) per flight; `cond.attrs['route_source']` says where the routes came from."""
    sheet = pd.read_excel(os.path.join(data_dir, FLIGHT_SHEET))
    sheet = sheet.dropna(subset=['Flight #'])
    cond = pd.DataFrame({
        'payload': sheet['Payload [g]'].values,
        'speed': sheet['Speed [m/s]'].values,
        'altitude': sheet['Altitude [m]'].astype(str).values,
        'route': sheet['Route #'].astype(str).values,
    }, index=sheet['Flight #'].astype(int).values)
    source = FLIGHT_SHEET

    path = fetch_kilthub_params(data_dir, download)
    routes = _kilthub_routes(path) if path else None
    if routes is not None:
        known = cond.index.isin(list(routes))
        if known.any():
            cond.loc[known, 'route'] = [routes[f] for f in cond.index[known]]
            source = KILTHUB_PARAMS
        else:
            warnings.warn('{} shares no flight numbers with the flight sheet; ignoring it'.format(path))
    if source == FLIGHT_SHEET:
        warnings.warn('routes come from the flight sheet, which mislabels some main-route flights; '
                      'run scripts/download_data.sh or retry with network access for the KiltHub routes')

    cond['main_route'] = cond['route'] == cond['route'].mode()[0]
    cond.attrs['route_source'] = source
    return cond


def _held_out(cond, split):
    """(test mask, excluded mask) over `cond`'s rows."""
    off_route = ~cond['main_route']
    if split == 'payload500':
        return cond['payload'] == 500, off_route
    if split == 'payload250':
        return cond['payload'] == 250, off_route
    if split == 'speed12':
        return cond['speed'] == 12, off_route
    if split == 'speed10-12':
        return cond['speed'] >= 10, off_route
    if split == 'speed8-12':
        return cond['speed'] >= 8, off_route
    if split == 'alt100':
        return cond['altitude'] == '100', off_route | cond['altitude'].str.contains('-')
    if split == 'route':
        return off_route, np.zeros(len(cond), dtype=bool)
    raise ValueError('unknown split {!r}, expected one of {}'.format(split, SPLITS))


def get_split(all_data, split='random', data_dir='./data/', val_split=0.25, random_seed=42, cond=None):
    """(test_range, val_range, train_range) arrays of flight numbers for `split`.

    cond: the output of `load_conditions`, to avoid re-reading it for every split.
    """
    if split == 'random':
        _, _, _, test_range, val_range, train_range = get_flights_and_ranges(all_data, val_split=val_split)
        return test_range, val_range, train_range

    if split in CONTROLS:
        test_range, val_range, train_full = get_split(all_data, 'random', data_dir, val_split, random_seed)
        if cond is None:
            cond = load_conditions(data_dir)
        n = len(get_split(all_data, CONTROLS[split], data_dir, val_split, random_seed, cond)[2])
        rng = np.random.RandomState(random_seed)
        train_range = np.sort(rng.choice(train_full, size=n, replace=False))
        return test_range, val_range, train_range

    if cond is None:
        cond = load_conditions(data_dir)
    cond = cond.loc[sorted(all_data)]
    test, excluded = _held_out(cond, split)
    unscorable = cond.index.isin(NO_MAPE_FLIGHTS)
    excluded = excluded | (test & unscorable)
    test = test & ~excluded
    test_range = cond.index[test].values
    pool = cond.index[~test & ~excluded].values

    rng = np.random.RandomState(random_seed)
    val_range = np.sort(rng.choice(pool, size=int(np.ceil(len(pool) * val_split)), replace=False))
    train_range = np.setdiff1d(pool, val_range)
    return test_range, val_range, train_range


if __name__ == '__main__':
    from data_utils import load_dataset
    all_data = load_dataset('./data/')
    cond = load_conditions('./data/')
    print('routes from', cond.attrs['route_source'], '| off-route flights:',
          list(cond.index[~cond['main_route'] & cond.index.isin(list(all_data))]))
    cond = cond.loc[sorted(all_data)]
    rows = []
    for split in SPLITS:
        test, val, train = get_split(all_data, split, cond=cond)
        t = cond.loc[np.unique(test)]
        rows.append({'split': split, 'train': len(train), 'val': len(val), 'test': len(np.unique(test)),
                     'test payloads (g)': sorted(t['payload'].unique().astype(int)),
                     'test speeds (m/s)': sorted(t['speed'].unique()),
                     'test altitudes (m)': sorted(t['altitude'].unique())})
    print(pd.DataFrame(rows).to_string(index=False))
