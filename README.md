# MTSP: Multi-Task Subscore Prediction for Depression Assessment

<p align="center">
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL%202026-Main-blue" alt="ACL 2026 Main"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-orange" alt="PyTorch">
</p>

<p align="center">
  <b>Rethinking Depression Prediction from a Fine-Grained Subscore Modeling Perspective via Multi-Task Learning</b><br>
  Zhenguang Wang, Bo Li, Wenhui Tan, Peng Cao†, Yang Wang, Jia Duan, Fei Wang†, Osmar Zaiane<br>
  <i>ACL 2026 Main</i>
</p>

Prior automated depression assessment methods predict only a single total score, missing fine-grained symptom-level information. **MTSP** reframes this as multi-task subscore prediction — jointly modeling all subscores with a Task Correlation Graph (GAT) and Task-Level Self-Paced Learning.

<p align="center">
  <img src="figures/architecture.png" width="85%" alt="MTSP Architecture">
</p>

## Results

| Model | Dev MAE↓ | Dev RMSE↓ | Test MAE↓ | Test RMSE↓ |
|:---|:---:|:---:|:---:|:---:|
| Qwen3-14B (Zero-Shot) | 5.43 | 5.49 | 5.39 | 5.47 |
| Ray et al. (2019) | — | 4.37 | 4.02 | 4.73 |
| Sadeghi et al. (2023) | 3.65 | 5.27 | 4.26 | 5.37 |
| Sadeghi et al. (2024) | 3.17 | 4.51 | 4.22 | 5.07 |
| Schmidt et al. (2025) s2o | 3.55 | 4.58 | 4.18 | 5.23 |
| Schmidt et al. (2025) s2s | 3.47 | 4.57 | 3.85 | **4.52** |
| **MTSP (Ours)** | **2.94** | **4.18** | **3.48** | 4.57 |

> State-of-the-art on E-DAIC (PHQ-8). **9.6%** MAE improvement over best prior work.

## Quick Start

```bash
git clone https://github.com/wxcwzg/MTSP.git && cd MTSP
pip install torch transformers numpy pandas tqdm tensorboard
```

```bash
# Full model (E-DAIC / PHQ-8)
python train_multi_scale.py --dataset edaic --scale PHQ-8 \
    --use_task_spl --use_cluster_constraint --use_task_graph

# CIDH / HAMD-13
python train_multi_scale.py --dataset cidh --scale HAMD-13 \
    --use_task_spl --use_cluster_constraint --use_task_graph

# PDCH / HAMD-13
python train_multi_scale.py --dataset pdch --scale HAMD-13 \
    --use_task_spl --use_cluster_constraint --use_task_graph
```

## Citation

```bibtex
@inproceedings{wang2026mtsp,
  title     = {Rethinking Depression Prediction from a Fine-Grained Subscore Modeling Perspective via Multi-Task Learning},
  author    = {Wang, Zhenguang and Li, Bo and Tan, Wenhui and Cao, Peng and Wang, Yang and Duan, Jia and Wang, Fei and Zaiane, Osmar},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## Acknowledgements

We thank the anonymous reviewers and the area chair for their constructive feedback. This work was supported by the National Natural Science Foundation of China (62076059, 82501861), the Science and Technology Joint Project of Liaoning Province (2023JH2/101700367), the Natural Science Foundation of Jiangsu Province (BK20240272), the Special Funds for Health Science and Technology Development of Nanjing Municipal Health Commission (YKK24188), and the General Project of the Science and Technology Development Foundation of Nanjing Medical University (NMUB20230205).
