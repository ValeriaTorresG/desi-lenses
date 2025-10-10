import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

HSC_SOURCE_URL = "https://www-utap.phys.s.u-tokyo.ac.jp/~oguri/sugohi/download/list_ra_asc_public.csv"
HSC_CHUNK_SIZE = 1 << 20
HSC_COLS = ['name','ra','dec','z_lens','z_source','zl_phot','zs_phot',
            'Rein','lens_mag_i','source_mag_i','type','discovery','grade','Reference']


def download_hsc_catalog(dest_path, url=HSC_SOURCE_URL, chunk_size=HSC_CHUNK_SIZE):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"----- HSC catalog not found. Downloading from {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
    if total:
        print(f"  Downloaded {downloaded} / {total} bytes.")
    else:
        print(f"  Downloaded {downloaded} bytes.")
    print(f"  Saved to {dest.resolve()}")
    return dest


def ensure_hsc_catalog(path):
    path = Path(path)
    if path.exists():
        return path
    return download_hsc_catalog(path)


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

        missing = [col for col in DESI_COLUMNS if col not in column_names]
        if missing:
            raise ValueError(f"Missing columns in DESI catalog: {', '.join(sorted(missing))}")

        ra = np.array(data["TARGET_RA"], dtype=np.float64)
        dec = np.array(data["TARGET_DEC"], dtype=np.float64)

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
                        default=Path("/pscratch/sd/v/vtorresg/desi-lenses/hsc_desi_matches.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    radius = args.radius_arcsec * u.arcsec

    hsc_path = ensure_hsc_catalog(args.hsc_csv)

    print(f"----- Reading HSC catalog {hsc_path}")
    lens_df, lens_coords = load_lens_catalog(hsc_path)
    print(f"  {len(lens_df)} lenses loaded out of {len(lens_df) + lens_df['RA'].isna().sum()} total entries.")

    print(f"----- Getting DESI coords from {args.desi_fits}")
    _, desi_coords = load_desi_coordinates(args.desi_fits)
    print(f"  {len(desi_coords)} entries in the DESI z cat")

    print(f"Searching in r<={radius.to_value(u.arcsec):.2f} arcsec")
    idx_desi, idx_lens, sep2d, _ = lens_coords.search_around_sky(desi_coords, radius)
    idx_lens = np.asarray(idx_lens, dtype=np.int64)
    idx_desi = np.asarray(idx_desi, dtype=np.int64)

    print(f"  {len(idx_desi)} found matches for {len(np.unique(idx_lens))} lenses.")
    if idx_lens.size:
        print(f"  Lens index stats -> min: {idx_lens.min()}, max: {idx_lens.max()}, len_df: {len(lens_df)}")

    valid = (idx_lens >= 0) & (idx_lens < len(lens_df))
    if not np.all(valid):
        invalid_count = np.count_nonzero(~valid)
        invalid_min = idx_lens[~valid].min()
        invalid_max = idx_lens[~valid].max()
        print(f"  WARNING: filtered {invalid_count} match(es) with invalid lens indices "
              f"(min {invalid_min}, max {invalid_max}).")
        idx_lens = idx_lens[valid]
        idx_desi = idx_desi[valid]
        sep2d = sep2d[valid]

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