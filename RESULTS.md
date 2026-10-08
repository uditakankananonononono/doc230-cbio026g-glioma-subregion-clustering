# RESULTS - 026-S (glioma subregions, BraTS 2024 post-treatment, 40 patients)

partial: no GPU, no CNN trained. Voxel k-means, a 30-tree random forest and a patch MLP only. Nothing here speaks to BRAINMAP-NET.

All numbers from results/results.json (python3 src/run.py, 205 s). Patient-level 5-fold CV, 20,000 sampled brain voxels per patient (800k total). Label fractions of sampled voxels: background 95.7%, NETC 0.11%, SNFH 2.9%, ET 0.43%, RC 0.83% (order 0,1,2,3,4 in results.json: 0.9568, 0.0011, 0.0295, 0.0043, 0.0083).

Prior-run context: an earlier run lost to a workspace rebuild had already shown K1 ARI 0.005 and K2 all Dice 0.0. This run reproduced them exactly; they are one result, not two. The compute amendment (30 trees, 200k-voxel training subsample) is in PREREG.md, committed before scoring.

| Dir | Result | Gate | Met |
|---|---|---|---|
| K1 k-means ARI | 0.0047 (shuffle control -0.00003) | >0.10 and >shuffle+0.05 | NO |
| K2 cluster->label Dice | ET 0, NETC 0, SNFH 0, RC 0 | ET > 0.30 | NO |
| K3 random forest, macro Dice | 0.233 (ET 0.403, NETC 0.0, SNFH 0.507, RC 0.023); label-shuffled RF 0.0 | > K2 macro | YES, but K2 macro is 0, so the bar is trivial |
| K4 patch MLP, macro Dice | 0.375 (ET 0.649, NETC 0.058, SNFH 0.661, RC 0.130) | > K3 + 0.02 | YES (+0.14) |
| K5 raw vs z-scored k-means | ARI 0.0108 vs 0.0047, delta 0.006 | abs(delta) < 0.05 | YES, but both are near zero |

Gates met: 3 of 5 on paper; the only substantive positives are the supervised ones (K3, K4). Honest reading: unsupervised intensity clustering does not recover glioma subregions. Supervised models recover ET and edema (SNFH) to a useful degree even from raw voxel/patch intensities; the small necrotic core (NETC) and resection cavity (RC) are not recovered. The patch MLP beating the 4-intensity RF is expected, since patches add local texture. Not a CNN, not full-volume segmentation.

Caveats: Dice is pooled over randomly sampled voxels, not full-volume segmentation Dice, so it is not comparable to published BraTS numbers; 40 patients; no per-fold CIs reported here (fold spread for K1 is in results.json); scanner/site intensity differences only partly handled by per-scan z-scoring; post-treatment anatomy.
