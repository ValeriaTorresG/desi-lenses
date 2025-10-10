import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True


def to_int_exact(x):
    import numpy as np
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(Decimal(str(x)))
    s = str(x).strip()
    if s == "":
        raise ValueError()
    try:
        return int(s)
    except Exception:
        try:
            return int(Decimal(s))
        except (InvalidOperation, ValueError) as e:
            raise ValueError() from e


def normalize_ids_array(ids):
    if ids.dtype.kind in ("i", "u"):
        return ids.astype(np.int64, copy=False)
    out = np.empty(ids.shape, dtype=np.int64)
    it = np.nditer(ids, flags=["multi_index", "refs_ok"])
    for v in it:
        out[it.multi_index] = to_int_exact(v.item())
    return out



class UmapCache:
    def __init__(self, umap_dir: str):
        self.umap_dir = umap_dir
        self.cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, night: int, tile: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (night, tile)
        if key in self.cache:
            return self.cache[key]
        path = Path(self.umap_dir) / f"umap_{night}_{tile}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Doesnt exist {path}")
        data = np.load(path, allow_pickle=True)
        arr = data["embedding"]
        mask = data["outlier_mask"]
        ids  = normalize_ids_array(data["ids"])
        if arr.shape[0] != ids.shape[0] or arr.shape[0] != mask.shape[0]:
            raise ValueError(f"embedding={arr.shape}, ids={ids.shape}, mask={mask.shape}")
        self.cache[key] = (arr, mask, ids)
        return self.cache[key]



def plot_umap_for_targets(arr, mask, ids, targets_with_types, night, tile, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"umap_{night}_{tile}_targets.png"

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.3, zorder=-1)

    ax.scatter(arr[:, 0], arr[:, 1],
               s=5, zorder=1, color='black', linewidths=0.0,
               edgecolor='none', alpha=0.10, label='All data')

    if mask.any():
        outliers = arr[mask]
        ax.scatter(outliers[:, 0], outliers[:, 1],
                   zorder=10, color='royalblue', s=2,
                   alpha=1.0, label='Outliers')

    missing = []
    id2type: Dict[int, str] = {}
    for t, ty in targets_with_types:
        t = int(t)
        if t not in id2type or (not id2type[t] and ty):
            id2type[t] = str(ty) if pd.notna(ty) else ""

    for target_id in sorted(id2type.keys()):
        type_str = id2type[target_id].strip() or "?"
        target_mask = (ids == target_id)
        if target_mask.any():
            target_coords = arr[target_mask]
            ax.scatter(target_coords[:, 0], target_coords[:, 1],
                       zorder=8, color='black', s=35,
                       marker='*', linewidths=1.0,
                       label=f'TARGETID={target_id} ({type_str})')
        else:
            missing.append(target_id)

    if missing:
        ax.text(0.02, 0.98,
                f"{len(missing)} TARGETID no está/n",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, color='crimson')

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Tile {tile} - Night {night}")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=360)
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/pscratch/sd/v/vtorresg/umap_analysis/data/resultados2.csv")
    parser.add_argument("--umap-dir", default="/pscratch/sd/v/vtorresg/umap_analysis/data/processed/umap")
    parser.add_argument("--outdir", default="./plots")
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--type-col", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.resultados, dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}

    reqs = ["targetid", "tileid", "night"]
    missing = [r for r in reqs if r not in cols_lower]
    if missing:
        raise ValueError()

    col_tid = cols_lower["targetid"]
    col_tile = cols_lower["tileid"]
    col_night = cols_lower["night"]

    if args.type_col:
        if args.type_col not in df.columns:
            raise ValueError()
        col_type = args.type_col
    else:
        cand = [c for c in df.columns if c.strip().lower() in ("type", "lens_type")]
        col_type = cand[0] if cand else None

    df["TARGETID_INT"] = df[col_tid].map(to_int_exact)
    df["TILEID_INT"] = df[col_tile].map(to_int_exact)
    df["NIGHT_INT"] = df[col_night].map(to_int_exact)

    groups: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for _, row in df.iterrows():
        night = int(row["NIGHT_INT"])
        tile = int(row["TILEID_INT"])
        tid = int(row["TARGETID_INT"])
        ttype = ""
        if col_type is not None:
            val = row[col_type]
            if pd.notna(val) and str(val).strip():
                ttype = str(val).strip()
        groups.setdefault((night, tile), []).append((tid, ttype))

    group_items = list(groups.items())
    if args.limit_groups is not None:
        group_items = group_items[:args.limit_groups]

    cache = UmapCache(args.umap_dir)
    outdir = Path(args.outdir)

    for (night, tile), targets_with_types in group_items:
        try:
            arr, mask, ids = cache.get(night, tile)
        except Exception as e:
            print(f"Error loading umap for night={night} tile={tile}: {e}")
            continue
        try:
            outpng = plot_umap_for_targets(arr, mask, ids, targets_with_types, night, tile, outdir)
            print(f"-Saved: {outpng} (targets={len(set(t for t,_ in targets_with_types))})")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()