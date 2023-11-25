# CEF-SsL
Repository for the paper "SoK: The Impact of Unlabelled Data in Cyberthreat Detection", published at EuroSP2022.

If you use any part of this codebase, you are kindly invited to cite our paper:

```
@inproceedings{apruzzese2022unlabelled,
  title={{SoK: The Impact of Unlabelled Data in Cyberthreat Detection}},
  author={Apruzzese, Giovanni and Laskov, Pavel and Tastemirova, Aliya},
  booktitle={7th IEEE European Symposium on Security and Privacy (EuroSP)},
  year={2022},
  organization={IEEE}
}
```



## Description
This repository contains two folders: "figures", which includes all figures (i.e., F1-score, Precision, Recall) related to the evaluation performed in the paper; and "code", which contains a prototype version of CEF-SsL, as well as a jupyter notebook showcasing its application in practice.

### Datasets
In our paper, we relied on 9 different datasets --- all of which are publicly available and can be obtained by following the links provided in the main paper.
The jupyter notebook provided in this repository uses a snippet of the "fixed" version of one of these datasets, the CIC-IDS17. For simplicity, such snippet only distinguishes between benign and malicious samples (all of which are randomly drawn from the "full" version of the troubleshooted CIC-IDS17).

### Disclaimer
For obvious reasons (e.g., copyright and space) we cannot include in this repo *all* the considered datasets used in our paper. In the (hopefully unlikely!) event that any of such dataset becomes unavailable, feel free to contact me (giovanni.apruzzese@uni.li). 
The data-snippet included in this repository has been authorized by the authors of the full dataset, whose complete details can be found at the following link: https://downloads.distrinet-research.be/WTMC2021/

Finally, we remark that the results provided in our paper are derived after hundreds (sometimes thousands) of trials. As such, single runs of CEF-SsL may provide different results than those reported in our paper. The only consistent result is the one of the "upper bound" SL, which assumes that all training data is 100% labelled.

## Updates
This repository is going to be incrementally updated. In particular, attention will be given to new findings derived from the usage of CEF-SsL. For instance, we have included in the "figures" folder the results on the (full) troubleshooted version of CICIDS17, and provided a description of such results in the dedicated "extension" document. 
Moreover, feel free to contact me if you make any new discoveries by using CEF-SsL: I will be glad to list all such discoveries in this page, so that you are given proper credit.

## Changelog
- June 12th, 2022: Added results on the troubleshooted version of CICIDS17
- June 11th, 2022: First code push and updated ReadMe
- March 2nd, 2022: Created repository with additional figures
