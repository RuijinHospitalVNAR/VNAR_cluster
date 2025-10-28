import subprocess
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    cifs = repo / 'examples' / 'cifs'
    # 1) download small sample
    subprocess.check_call(['python', str(repo / 'examples' / 'download_cifs.py'), '--out_dir', str(cifs)])
    # 2) run clustering
    out = repo / 'examples' / 'results'
    cmd = [
        'python', str(repo / 'cluster_interfaces.py'),
        '--cif_dir', str(cifs),
        '--antibody_chain', 'A',
        '--antigen_chains', 'B',
        '--cutoff', '5.0',
        '--method', 'hdbscan',
        '--out_dir', str(out),
        '--save_plots', '--save_numpy'
    ]
    subprocess.check_call(cmd)
    print(f"Done. See results in: {out}")


if __name__ == '__main__':
    main()
