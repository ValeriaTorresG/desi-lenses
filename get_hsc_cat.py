import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

HSC_COLS = ['name','ra','dec','z_lens','z_source','zl_phot','zs_phot',
            'Rein','lens_mag_i','source_mag_i','type','discovery','grade','Reference']


def load_lens_catalog(path):
    df = pd.read_csv(path, header=None, names=HSC_COLS)
    if df.empty:
        raise ValueError(f"hsc cat is empty: {path}")

    # las columnas se llaman distinto en los dos archivos
    columns_lower = {col.lower(): col for col in df.columns}
    if "ra" not in columns_lower or "dec" not in columns_lower:
        raise ValueError()

    ra_col = columns_lower["ra"]
    dec_col = columns_lower["dec"]

    df = df.dropna(subset=[ra_col, dec_col]).copy()
    if df.empty:
        raise ValueError()

    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "lens_row_index"}, inplace=True)
    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    df.rename(columns=str.upper, inplace=True)

    coords = SkyCoord(ra=np.asarray(df["RA"], dtype=float) * u.deg,
                      dec=np.asarray(df["DEC"], dtype=float) * u.deg)
    return df, coords


DESI_COLUMNS = ["TARGETID",
                "HEALPIX",
                "SURVEY",
                "Z",
                "ZERR",
                "ZWARN",
                "SPECTYPE",
                "SUBTYPE",
                "TARGET_RA",
                "TARGET_DEC",
                "OBJTYPE",
                "MEAN_FIBER_RA",
                "MEAN_FIBER_DEC",
                "DESI_TARGET",
                "BGS_TARGET",
                "MWS_TARGET",
                "SCND_TARGET"] #? tal vez no necesite tener las de target


def load_desi_coordinates(path):
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError()

        data = hdul[1].data

        column_names = set(data.columns.names)
        coord_required = {"RA", "DEC"}
        if not coord_required.issubset(column_names):
            missing_coords = coord_required - column_names
            raise ValueError(f"Missing coordinates: {', '.join(sorted(missing_coords))}")

        missing = [col for col in DESI_COLUMNS if col not in column_names]
        if missing:
            raise ValueError(f"Missing columns in DESI catalog: {', '.join(sorted(missing))}")

        ra = np.array(data["RA"], dtype=np.float64)
        dec = np.array(data["DEC"], dtype=np.float64)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    return ra, coords


def gather_desi_rows(path, indices):
    if indices.size == 0:
        return pd.DataFrame()

    with fits.open(path, memmap=True) as hdul: #revisar el memmap, el archivo es de ~20gb
        data = hdul[1].data
        subset = data[indices]
        table = Table(subset)[DESI_COLUMNS]
        df = table.to_pandas()

    df.columns = [str(c).upper() for c in df.columns]
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsc-csv", type=Path,
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/hsc_full_catalog.csv"))
    parser.add_argument("--desi-fits", type=Path,
                        default=Path("/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits"))
    parser.add_argument("--radius-arcsec", type=float,
                        default=6.0) #* ESTE ES EL r DEL PAPER DE https://arxiv.org/abs/2505.16158v2
    parser.add_argument("--output", type=Path,
                        default=Path("hsc_desi_matches.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    radius = args.radius_arcsec * u.arcsec

    print(f"----- Reading HSC catalog {args.hsc_csv}")
    lens_df, lens_coords = load_lens_catalog(args.hsc_csv)
    print(f"  {len(lens_df)} lenses loaded out of {len(lens_df) + lens_df['RA'].isna().sum()} total entries.")

    print(f"----- Getting DESI coords from {args.desi_fits}")
    _, desi_coords = load_desi_coordinates(args.desi_fits)
    print(f"  {len(desi_coords)} entries in the DESI z cat")

    print(f"Searching in r<={radius.to_value(u.arcsec):.2f} arcsec")
    idx_desi, idx_lens, sep2d, _ = desi_coords.search_around_sky(lens_coords, radius)
    print(f"  {len(idx_desi)} found matches for {len(np.unique(idx_lens))} lenses.")

    print("----- Getting matched rows and saving")
    desi_matches_df = gather_desi_rows(args.desi_fits, idx_desi)
    lens_matches_df = lens_df.iloc[idx_lens].reset_index(drop=True)

    matches = pd.concat([desi_matches_df.reset_index(drop=True),
                         lens_matches_df.reset_index(drop=True)],
                        axis=1)
    matches["SEPARATION_ARCSEC"] = sep2d.to(u.arcsec).value

    matches.columns = [str(c).upper() for c in matches.columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"----- Saved in {args.output.resolve()}")


if __name__ == "__main__":
    main()