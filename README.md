partial: voxel k-means, random forest and patch MLP on 40 BraTS-2024 post-treatment glioma patients; no CNN trained and no claim about the CBIO026-2026 BRAINMAP-NET core.

See PREREG.md (committed first, incl. a compute amendment before scoring), RESULTS.md, results/results.json, src/run.py, data_manifest.json. Data (not included): HF YongchengYAO/BraTS24-Lite (CC BY-NC 4.0), remote-unzipped by the sampling rule in the prereg.

## Reproduce / readability audit (2026-10-09)
Data: BraTS24-GLI (HF dataset YongchengYAO/BraTS24-Lite) for 026g, Davis from DeepDTA GitHub for 028x. Scripts read /tmp/brats/ and /tmp/davis/ respectively; no SHA256 manifest in this repo (gap). Run python3 src/run.py.
