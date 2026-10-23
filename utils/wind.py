"""Wind from the KiltHub logs (data/kilthub/flights/<flight>.csv): anemometer airflow + GNSS/INS ground velocity.

Conventions, checked against the data (`python -m utils.wind --check`):
    velocity_x/y      body frame: y forward, x right (y ~ commanded speed on every leg; differentiating lon/lat
                      overestimates the ground speed by ~20 %, so position is not used)
    orientation_*     north-west-up quaternion: compass heading = -yaw (matches the GNSS track to ~1 deg)
    wind_speed/angle  airflow past the drone; angle = compass direction the air comes FROM, w.r.t. magnetic north
so the wind (east, north) in m/s is

    w = v_ground - s * (sin(a), cos(a)),    a = wind_angle + DECLINATION + offset

DECLINATION is the WMM2015v2 declination at the takeoff point in 2019 (flights were flown in Penn Hills, PA,
Rodrigues et al. 2021; takeoff 40.459 N, 79.783 W): -9.4 deg, constant to 0.01 deg over Apr-Oct 2019.
`offset` is any further misalignment of the anemometer. It is 0 by default: over the main-route flights the
declination lowers the median leg gap from 6.2 to 4.5 m/s, and a further -6 deg only to 4.2 m/s, with the best
value varying by date between about -21 and 0 deg (see --check).

The wind is only meaningful in cruise (prop wash and low airspeed dominate in climb / hover / descent), so a
flight's wind is the mean over its cruise steps: ground speed > 70 % of the commanded speed, > 15 m above takeoff.
`leg_gap` (largest difference between the mean wind of the route's legs) estimates how well the wind is resolved;
it is ~3-4 m/s, so only the flight-mean wind is reliable, not the per-step wind.

    python -m utils.wind              # per-flight wind -> data/kilthub/wind.csv, summary by date
    python -m utils.wind --check      # the convention / offset checks behind the formula above
"""
import argparse
import os

import numpy as np
import pandas as pd

DECLINATION = -9.4  # deg, magnetic -> true north (west is negative)
KILTHUB_DIR = os.path.join('data', 'kilthub')


def unit(deg):
    """Compass degrees (from north, clockwise) -> unit vectors (east, north)."""
    r = np.deg2rad(deg)
    return np.stack([np.sin(r), np.cos(r)], -1)


def load_params(kdir=KILTHUB_DIR):
    return pd.read_csv(os.path.join(kdir, 'parameters.csv')).set_index('flight')


def load_flight(flight, kdir=KILTHUB_DIR):
    return pd.read_csv(os.path.join(kdir, 'flights', '{}.csv'.format(flight)))


def heading(d):
    """Compass heading (deg) of the nose, from the north-west-up quaternion."""
    qx, qy, qz, qw = (d['orientation_' + c].values for c in 'xyzw')
    return -np.rad2deg(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))


def ground_velocity(d):
    """(n, 2) ground velocity (east, north) in m/s: body-frame velocity rotated by the heading."""
    h = heading(d)
    return d['velocity_y'].values[:, None] * unit(h) + d['velocity_x'].values[:, None] * unit(h + 90)


def cruise_mask(d, vg, speed):
    """Steps in cruise: ground speed > 70 % of the commanded speed and > 15 m above takeoff."""
    fix = d['position_y'].values != 0  # some logs start before the GNSS fix (lat = lon = alt = 0)
    if not fix.any():  # ground tests that never get a fix
        return fix
    up = d['position_z'].values - d['position_z'].values[fix][:20].mean()
    return fix & (np.hypot(*vg.T) > 0.7 * speed) & (up > 15)


def airflow(d, offset=0., sense='from', frame='north'):
    """(n, 2) airflow past the drone (east, north): where the air goes, relative to the drone.

    sense / frame are the alternatives tested in `check`: 'toward' if wind_angle were the direction the air goes,
    'nose' if it were relative to the nose instead of magnetic north.
    """
    a = d['wind_angle'].values + DECLINATION + offset
    if frame == 'nose':
        a = d['wind_angle'].values + heading(d) + offset
    flow = d['wind_speed'].values[:, None] * unit(a)
    return -flow if sense == 'from' else flow


def wind(d, offset=0., **kwargs):
    """(n, 2) wind (east, north) in m/s at every step; see the module docstring for when it is meaningful."""
    return ground_velocity(d) + airflow(d, offset, **kwargs)


def leg_gap(w, vg, min_steps=10):
    """Largest distance between the mean wind of the legs (steps grouped by ground track into 90 deg sectors)."""
    sector = ((np.rad2deg(np.arctan2(vg[:, 0], vg[:, 1])) + 45) % 360 // 90).astype(int)
    legs = [w[sector == q].mean(0) for q in range(4) if (sector == q).sum() >= min_steps]
    return max(np.linalg.norm(p - q) for p in legs for q in legs) if len(legs) > 1 else np.nan


def flight_wind(flight, speed, kdir=KILTHUB_DIR, offset=0., **kwargs):
    """Cruise-mean wind of one flight: speed (m/s), direction it comes from (deg), leg gap (m/s), cruise steps."""
    d = load_flight(flight, kdir)
    vg = ground_velocity(d)
    cruise = cruise_mask(d, vg, speed)
    if cruise.sum() < 30:  # hover / ground tests
        return {'wind': np.nan, 'wind_dir': np.nan, 'leg_gap': np.nan, 'n_cruise': int(cruise.sum())}
    w = vg[cruise] + airflow(d, offset, **kwargs)[cruise]
    m = w.mean(0)
    return {'wind': np.linalg.norm(m), 'wind_dir': np.rad2deg(np.arctan2(-m[0], -m[1])) % 360,
            'leg_gap': leg_gap(w, vg[cruise]), 'n_cruise': int(cruise.sum())}


def load_wind(kdir=KILTHUB_DIR, offset=0., flights=None):
    """Per-flight cruise wind for every flight in parameters.csv (or `flights`), indexed by flight."""
    params = load_params(kdir)
    flights = params.index if flights is None else flights
    rows = {f: flight_wind(f, params.loc[f, 'speed'], kdir, offset) for f in flights}
    return pd.DataFrame.from_dict(rows, orient='index').rename_axis('flight')


def check(kdir=KILTHUB_DIR):
    """Print the evidence for the conventions: speed / heading consistency, the four readings of wind_angle,
    and the angle offset (beyond the declination) that best removes the leg gap, overall and per date."""
    params = load_params(kdir)
    main = params[params['route'] == params['route'].mode()[0]]
    sample = [f for s in sorted(main['speed'].unique()) for f in main[main['speed'] == s].index[:3]]

    rows = []
    for f in sample:
        d = load_flight(f, kdir)
        vg = ground_velocity(d)
        cruise = cruise_mask(d, vg, main.loc[f, 'speed'])
        lat = np.deg2rad(d['position_y'][d['position_y'] != 0].mean())
        sm = lambda x: pd.Series(x).rolling(5, center=True, min_periods=1).mean().values
        east = np.gradient(sm(d['position_x'].values * 111320 * np.cos(lat)), d['time'])
        north = np.gradient(sm(d['position_y'].values * 111034), d['time'])
        track = np.rad2deg(np.arctan2(east, north))
        r = {'flight': f, 'cmd speed': main.loc[f, 'speed'],
             'body speed': np.median(np.hypot(*vg[cruise].T)),
             'position speed': np.median(np.hypot(east, north)[cruise]),
             'heading - track': np.median(((heading(d) - np.rad2deg(np.arctan2(*vg.T)) + 180) % 360 - 180)[cruise]),
             'velocity dir - track': np.median(((np.rad2deg(np.arctan2(*vg.T)) - track + 180) % 360 - 180)[cruise])}
        for frame in ['north', 'nose']:
            for sense in ['from', 'toward']:
                w = wind(d, frame=frame, sense=sense)[cruise]
                r['{},{} gap'.format(frame, sense)] = leg_gap(w, vg[cruise])
        rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('Ground speed and heading (median over cruise; body velocity rotated by heading vs differentiated lon/lat):')
    print(out.iloc[:, :6].round(2).to_string(index=False))
    print('\nMedian leg gap (m/s) for each reading of wind_angle, declination applied where north-referenced:')
    print(out[[c for c in out.columns if c.endswith('gap')]].median().round(2).to_string())

    offsets = np.arange(-30, 21, 2)
    rows = []
    for f in main.index:
        d = load_flight(f, kdir)
        vg = ground_velocity(d)
        cruise = cruise_mask(d, vg, main.loc[f, 'speed'])
        gaps = [leg_gap(wind(d, o)[cruise], vg[cruise]) for o in offsets]
        rows.append({'flight': f, 'date': main.loc[f, 'date'], 'gap@0': gaps[list(offsets).index(0)],
                     'best offset': offsets[int(np.nanargmin(gaps))], 'gap@best': np.nanmin(gaps), 'gaps': gaps})
    out = pd.DataFrame(rows)
    sweep = np.nanmedian(np.stack(out['gaps']), 0)
    print('\nMedian leg gap over all {} main-route flights vs extra offset (deg, on top of declination {}):'.format(
        len(out), DECLINATION))
    print('  ' + '  '.join('{:+d}:{:.2f}'.format(o, g) for o, g in zip(offsets, sweep)))
    print('\nPer-flight best extra offset, by date:')
    print(out.groupby('date').agg(n=('flight', 'size'), best_offset=('best offset', 'median'),
                                  gap_at_0=('gap@0', 'median'), gap_at_best=('gap@best', 'median')).round(1).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--check', action='store_true', help='Print the convention / offset checks.')
    parser.add_argument('--offset', type=float, default=0., help='Extra anemometer angle offset (deg).')
    args = parser.parse_args()

    kdir = os.path.join(args.directory, KILTHUB_DIR)
    if args.check:
        check(kdir)
    else:
        params = load_params(kdir)
        w = load_wind(kdir, args.offset).join(params[['speed', 'payload', 'altitude', 'date', 'route']])
        w.to_csv(os.path.join(kdir, 'wind.csv'))
        print(w.groupby('date')[['wind', 'leg_gap']].median().round(2).to_string())
        main = w[w['route'] == w['route'].mode()[0]]
        print('\nmain route: {} flights, wind median {:.2f} m/s, 80th percentile {:.2f} m/s, leg gap median {:.2f}'.format(
            len(main), main['wind'].median(), main['wind'].quantile(0.8), main['leg_gap'].median()))
        print('saved', os.path.join(kdir, 'wind.csv'))
