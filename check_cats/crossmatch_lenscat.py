import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from get_hsc_cat import gather_desi_rows, load_desi_coordinates


def load_lenscat_catalog(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"catalog not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"catalog is empty: {path}")

    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    columns_lower = {col.lower(): col for col in df.columns}

    def find_column(candidates):
        for cand in candidates:
            if cand in columns_lower:
                return columns_lower[cand]
        raise ValueError(f"Tried {candidates} in {df.columns}")

    ra_col = find_column(["ra [deg]", "ra", "ra_deg", "ra(deg)", "ra_deg"])
    dec_col = find_column(["dec [deg]", "dec", "dec_deg", "dec(deg)", "dec_deg"])

    df["RA"] = pd.to_numeric(df[ra_col], errors="coerce")
    df["DEC"] = pd.to_numeric(df[dec_col], errors="coerce")
    df = df.dropna(subset=["RA", "DEC"]).copy()
    if df.empty:
        raise ValueError("No entries remain after dropping rows without RA/DEC.")

    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "lenscat_row_index"}, inplace=True)
    df.rename(columns=str.upper, inplace=True)

    coords = SkyCoord(ra=np.asarray(df["RA"], dtype=np.float64) * u.deg,
                      dec=np.asarray(df["DEC"], dtype=np.float64) * u.deg)
    return df, coords


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lenscat-csv", type=Path,
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/lenscat_catalog.csv"))
    parser.add_argument("--desi-fits", type=Path,
                        default=Path("/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits"))
    parser.add_argument("--radius-arcsec", type=float,
                        default=6.0)
    parser.add_argument("--output", type=Path,
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/lenscat_desi_matches.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    radius = args.radius_arcsec * u.arcsec

    print(f"----- Reading lenscat catalog {args.lenscat_csv}")
    lenscat_df, lenscat_coords = load_lenscat_catalog(args.lenscat_csv)
    print(f"  {len(lenscat_df)} entries loaded.")

    print(f"----- Getting DESI coords from {args.desi_fits}")
    _, desi_coords = load_desi_coordinates(args.desi_fits)
    print(f"  {len(desi_coords)} entries in the DESI z catalog.")

    print(f"Searching in r<={radius.to_value(u.arcsec):.2f} arcsec")
    idx_desi, idx_lenscat, sep2d, _ = lenscat_coords.search_around_sky(desi_coords, radius)
    idx_lenscat = np.asarray(idx_lenscat, dtype=np.int64)
    idx_desi = np.asarray(idx_desi, dtype=np.int64)

    print(f"  {len(idx_desi)} found matches for {len(np.unique(idx_lenscat))} lenscat entries.")
    if idx_lenscat.size:
        print(f"  lenscat index stats -> min: {idx_lenscat.min()}, max: {idx_lenscat.max()}, len_df: {len(lenscat_df)}")

    valid = (idx_lenscat >= 0) & (idx_lenscat < len(lenscat_df))
    if not np.all(valid):
        invalid_count = np.count_nonzero(~valid)
        invalid_min = idx_lenscat[~valid].min()
        invalid_max = idx_lenscat[~valid].max()
        print(f"  WARNING: filtered {invalid_count} match(es) with invalid lenscat indices "
              f"(min {invalid_min}, max {invalid_max}).")
        idx_lenscat = idx_lenscat[valid]
        idx_desi = idx_desi[valid]
        sep2d = sep2d[valid]

    print("----- Getting matched rows and saving")
    desi_matches_df = gather_desi_rows(args.desi_fits, idx_desi)
    lenscat_matches_df = lenscat_df.iloc[idx_lenscat].reset_index(drop=True)

    matches = pd.concat([desi_matches_df.reset_index(drop=True),
                         lenscat_matches_df.reset_index(drop=True)], axis=1)
    matches["SEPARATION_ARCSEC"] = sep2d.to(u.arcsec).value

    matches.columns = [str(c).upper() for c in matches.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"----- Saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()