import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from get_hsc_cat import gather_desi_rows, load_desi_coordinates


def load_kids_catalog(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"catalog not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"catalog is empty: {path}")

    columns_lower = {col.lower(): col for col in df.columns}
    if "ra" not in columns_lower or "dec" not in columns_lower:
        raise ValueError(f"found: {sorted(df.columns)} not RA/DEC columns in catalog.")

    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "kids_row_index"}, inplace=True)
    df.rename(columns=str.upper, inplace=True)

    if "NAME" not in df.columns:
        raise ValueError("KIDS catalog must contain a NAME column.")
    if "KIDS_NAME" not in df.columns:
        df["KIDS_NAME"] = df["NAME"]

    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["DEC"] = pd.to_numeric(df["DEC"], errors="coerce")
    df = df.dropna(subset=["RA", "DEC"]).copy()
    if df.empty:
        raise ValueError("No KIDS entries remain after dropping rows without RA/DEC.")

    coords = SkyCoord(ra=np.asarray(df["RA"], dtype=np.float64) * u.deg,
                      dec=np.asarray(df["DEC"], dtype=np.float64) * u.deg)
    return df, coords


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kids-csv", type=Path,
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/kids_qso.csv"))
    parser.add_argument("--desi-fits", type=Path,
                        default=Path("/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits"))
    parser.add_argument("--radius-arcsec", type=float, default=6.0)
    parser.add_argument("--output", type=Path,
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/kids_desi_matches.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    radius = args.radius_arcsec * u.arcsec

    print(f"----- Reading KIDS catalog {args.kids_csv}")
    kids_df, kids_coords = load_kids_catalog(args.kids_csv)
    print(f"  {len(kids_df)} candidates loaded.")

    print(f"----- Getting DESI coords from {args.desi_fits}")
    _, desi_coords = load_desi_coordinates(args.desi_fits)
    print(f"  {len(desi_coords)} entries in the DESI z catalog.")

    print(f"Searching in r<={radius.to_value(u.arcsec):.2f} arcsec")
    idx_desi, idx_kids, sep2d, _ = kids_coords.search_around_sky(desi_coords, radius)
    idx_kids = np.asarray(idx_kids, dtype=np.int64)
    idx_desi = np.asarray(idx_desi, dtype=np.int64)

    print(f"  {len(idx_desi)} found matches for {len(np.unique(idx_kids))} KIDS entries.")
    if idx_kids.size:
        print(f"  KIDS index stats -> min: {idx_kids.min()}, max: {idx_kids.max()}, len_df: {len(kids_df)}")

    valid = (idx_kids >= 0) & (idx_kids < len(kids_df))
    if not np.all(valid):
        invalid_count = np.count_nonzero(~valid)
        invalid_min = idx_kids[~valid].min()
        invalid_max = idx_kids[~valid].max()
        print(f"  -- filtered {invalid_count} match(es) with invalid KIDS indices "
              f"(min {invalid_min}, max {invalid_max}).")
        idx_kids = idx_kids[valid]
        idx_desi = idx_desi[valid]
        sep2d = sep2d[valid]

    print("----- Getting matched rows and saving")
    desi_matches_df = gather_desi_rows(args.desi_fits, idx_desi)
    kids_matches_df = kids_df.iloc[idx_kids].reset_index(drop=True)

    matches = pd.concat([desi_matches_df.reset_index(drop=True), kids_matches_df.reset_index(drop=True)],
                        axis=1)
    matches["SEPARATION_ARCSEC"] = sep2d.to(u.arcsec).value

    matches.columns = [str(c).upper() for c in matches.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"----- Saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()