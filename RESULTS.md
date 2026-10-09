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

## Extension K6-K10 (prereg committed before scoring; single seed, 5-fold patient-grouped, 200k sampled training voxels per fold, Dice on 800k sampled voxels from 40 cases)
Reference RF (K3 config re-fit): macro Dice 0.2333 (ET 0.403, NETC 0, SNFH 0.507, RC 0.023); reproduces original K3 macro.
- K6 per-patient ET Dice (n=13 patients with >=50 ET voxels): median 0.330 (q25 0.245, q75 0.489). Gate (>0.20) MET.
- K7 modality ablation, ET Dice: drop t1n 0.344, t1c 0.003, t2w 0.447, t2f 0.002. Dropping t1c lowers ET by 0.40 (>=0.03). Gate MET.
- K8 k-means ARI vs labels: k=3 0.0033, k=5 0.0047, k=8 0.0146. Gate (best ARI >0.10, as in PREREG) NOT MET.
- K9 5x5 in-plane patch mean/std features added: macro 0.375 vs 0.233 (ET 0.634, NETC 0.058, SNFH 0.630, RC 0.180). Gate (+0.02) MET. Caveat: sampled voxels, single seed, patches clipped at edges.
- K10 class_weight=balanced_subsample: macro 0.227 vs 0.233. Gate (+0.02) NOT MET.
Tally: K6, K7, K9 met; K8, K10 not met.
Incident: src/ext_run.py ended silently after K8; part 2 (src/ext_run_part2.py) first died with exit 137 (OOM kill, no traceback) while building patch features; fixed by computing patch mean/std per case instead of holding the 100-column array. Logs: results/ext_run_part1.log, ext_run_part2.log.

### Gate review corrections (2026-10-09; wording only, tally unchanged at 3/5)
- K7: dropping t2f also collapses ET to 0.002, SNFH to 0.0 and macro to 0.001. Read this as a likely degenerate RF (all-class collapse), not a clean ablation of t2f. Effect sizes are single seed with a fresh voxel subsample per ablation, so they are not precise. The t1c drop (ET 0.403 to 0.003) is not the same all-class collapse: SNFH stays 0.479 and macro 0.127, so it is a large effect but not shown to be degenerate.
- K9: macro 0.375 beats K3 partly through neighbourhood information, and ET 0.634 does not exceed the K4 MLP (ET 0.649). Do not read it as new. Dice is pooled over sampled voxels, not full-volume Dice.
- K10: flat macro hides a per-class trade: SNFH 0.507 to 0.312, RC 0.023 to 0.167, NETC 0 to 0.038, ET 0.403 to 0.392.
- K6-K8 were produced by part 1, which died before writing JSON; the tally for them traces to results/ext_run_part1.log. results/ext_results.json is RECONSTRUCTED from that log (K6-K8, parsed verbatim) plus part 2's JSON (K9, K10).
- K8 gate wording aligned to the prereg (ARI > 0.10).
