$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
python "$repo\examples\download_cifs.py" --out_dir "$repo\examples\cifs"
python "$repo\two_stage_interfaces.py" --cif_dir "$repo\examples\cifs" --antibody_chain A --antigen_chains B --cutoff 5.0 --k_coarse 5 --min_cluster_size 5 --out_dir "$repo\examples\results" --save_plots --save_numpy
Write-Host "Done. Results in $repo\examples\results"
