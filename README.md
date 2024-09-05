# A multi-task framework for hyperspectral change detection and band reweighting with unbalanced contrastive learning. [IEEE Trans. Geosci. Remote. Sens.  (2024)]
This is our official implementation of MHCD!

by Xiande Wu, Paolo Gamba, Jie Feng, Ronghua Shang, Xiangrong Zhang, Licheng Jiao

# Introduction
## Abstract
Multi-task learning has been widely applied in visual learning to significantly enhance performances. The combination of hyperspectral change detection (HCD) and band selection can achieve discriminative feature selection for improving detection performance. However, existing multi-task models for these two tasks are unidirectional, with band selection unable to learn from task guidance. To address this challenge, a multi-task HCD framework with differential band selection and unbalanced contrastive learning (MHCD) is proposed. MHCD consists of a differential band selection network (DBSN) and a Siamese detection network. DBSN selects discriminative bands for HCD by analyzing the differential spatial-spectral information across time states, whose optimization is under the guidance of HCD. Furthermore, a multi-temporal interaction module and multi-domain fusion module are inserted into the Siamese detection network. They hierarchically connect cross-temporal features and fuse features from spatial, spectral, and temporal domains, providing complementary clues in these different domains. Considering the sample imbalance and enormous variation within a class in binary HCD, an unbalanced contrastive learning method based on multiple prototypes is tailored has been considered. It estimates multiple prototypes to flexibly adjust the contribution of different classes of samples to the loss. The proposed method has been validated using three public benchmark datasets, demonstrating improvements in multiple metrics for change detection.

## Figure.1 Flowchart of the proposed method. 
![image]([https://github.com/jiefeng0109/DMOEAD-main/blob/main/1725504747389.jpg](https://github.com/jiefeng0109/hyperspectral-change-detection-and-band-reweighting/blob/main/1725505201929.jpg))

## 目录

- [数据集描述](#a-namedatasetsa-)
- [用法](#a-nameusagea-)
    - [训练](#a-nameusage-traina-)
    - [测试](#a-nameusage-testa-)
- [试验记录](#a-nameresulta-)


## <a name="datasets"></a> 数据集描述

数据集来自 [BayArea, Hermiston & Farmland]

## <a name="usage"></a> 用法

### <a name="usage-train"></a> 训练

1. 运行main_contrastive.py文件

### <a name="usage-test"></a> 测试

验证集等于测试集，无需再另行测试


## Cite
```
@article{wu2024multi,
  title={A multi-task framework for hyperspectral change detection and band reweighting with unbalanced contrastive learning},
  author={Wu, Xiande and Gamba, Paolo and Feng, Jie and Shang, Ronghua and Zhang, Xiangrong and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}

```
