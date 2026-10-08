# PREREG - CBIO026 (2026 title: BRAINMAP-NET, CNN clustering glioma subregions), CPU slice "026-S"

partial: no GPU, so no full-resolution 3D CNN is trained. This slice tests only whether simple voxel-intensity clustering and small CPU classifiers recover expert glioma subregions on a small patient sample. No claim is made about BRAINMAP-NET.

Note: this is the 2026 CBIO026 (glioma). The 2023 CBIO026 (microbots) is a different parent.

## History and amendment (written before any score of this run existed)
A first run was started with the original config (RF 100 trees depth 12 on all ~640k training voxels per fold). Its workspace was lost mid-run (infrastructure rebuild), and the RF stage alone had been running over 25 minutes on 2 CPUs. Only K1 and K2 finished; their results (K1 ARI 0.005, shuffle -0.00003, gate 0.10 not met; K2 all four Dice 0.0, ET gate 0.30 not met) were reported to the coordinator as an interim result and are carried here as prior-run context, not re-counted. The repo and prereg of that run were lost with it; this prereg is a faithful rewrite plus the amendment below. The same sampling rule and seed are used, so K1/K2 should reproduce; if they differ it will be reported.
AMENDMENT: supervised models (K3, K4) are trained on a random 200,000-voxel subsample of each training fold (seed 0), and the random forest uses 30 trees (depth 12). Reason: compute.

## Data (real, public, CC BY-NC 4.0)
BraTS 2024 adult glioma post-treatment (GLI) via HF dataset YongchengYAO/BraTS24-Lite, remote-unzipped (only chosen files fetched).
Labels: 1 NETC, 2 SNFH, 3 ET, 4 RC (resection cavity). 4 modalities t1n,t1c,t2w,t2f.
Sample rule (fixed before looking at any image): list patients sorted by ID (731 patients, 1621 scans), take every 18th patient (first 40), use each patient's first sorted timepoint. 40 patients. Hashes in data_manifest.json.

## Preprocessing (fixed)
Brain mask = voxels nonzero in all 4 modalities. Per-scan z-score of each modality inside the brain mask. 20000 random brain voxels per scan (seed 0) used for everything below.
Split: patient-level 5-fold CV (patient index mod 5).

## Directions and gates (held-out patients; metrics pooled over sampled test voxels)
K1 Unsupervised k-means (k=5, fit per fold on training voxels, assign test voxels). Adjusted Rand index vs 5-class labels (0 + 4). Gate: ARI > 0.10 AND beats a within-patient label-shuffle control by >0.05.
K2 Majority-label mapping of clusters from training voxels; per-label Dice (ET, NETC, SNFH, RC). Gate: ET Dice > 0.30.
K3 Random forest on 4 intensities, patient-held-out; macro Dice over the 4 labels. Gate: macro Dice > K2 macro Dice. Shuffle control: trained on within-patient-permuted labels.
K4 Patch MLP (scikit-learn MLP 64-32 on 5x5 in-slice patches x 4 modalities = 100 features; a CPU stand-in, NOT a CNN). Gate: macro Dice > K3 + 0.02.
K5 K1 repeated with raw (not z-scored) intensities. Gate: |ARI change| < 0.05.

## Caveats
Intensities are not standardized across sites; z-score per scan mitigates only partly. 40 patients; fold spread only. Labels are very imbalanced (background ~96% of sampled brain voxels). Post-treatment anatomy differs from pre-operative BraTS. No gate is changed after results; misses are reported as misses.
