import argparse
from pathlib import Path
import requests

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def download_cif(pdb_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    path = out_dir / f"{pdb_id.upper()}.cif"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_ids', nargs='+', default=['1CBN','1BNA','1CRN'])
    parser.add_argument('--out_dir', type=str, default='examples/cifs')
    args = parser.parse_args()
    out = Path(args.out_dir)
    for pid in args.pdb_ids:
        p = download_cif(pid, out)
        print(f"Downloaded: {p}")


if __name__ == '__main__':
    main()
