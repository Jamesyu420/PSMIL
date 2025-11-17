# Detecting Breast Carcinoma Metastasis on Whole-Slide Images by Partially Subsampled Multiple Instance Learning

This repository provides reproducible codes for the following working paper:

> Yu, B., Li, X., Zhou, J., and Wang, H. Detecting Breast Carcinoma Metastasis on Whole-Slide Images by Partially Subsampled Multiple Instance Learning, Working Paper.

For an easy application of the proposed PSMIL method in this paper, you may also call the [PSMIL](https://test.pypi.org/project/PSMIL/) package.

## Overview

This repository contains three parts: the first covers simulation studies to validate the finite sample performance of the proposed estimators, the second focuses on robustness analysis, and the third presents real data analysis on the CAMELYON16 dataset.

The file tree of the repository is as follows:

```
├── LICENSE
├── README.md
├── RealDataAnalysis
├── Robustness
├── Simulation
├── datadict.txt
└── requirements.txt
```

- `datadict.txt` contains the data dictionary for the database.
- `requirements.txt` lists the specific Python package versions used in this repository.

## PART 1. Simulation Studies for Theoretical Properties

The following files in the `Simulation` folder can be used to reproduce the simulation results for the proposed estimators presented in the paper.

```
├── Simulation
│   ├── para_est_KmeansInsInit.pkl
│   ├── simucode_Study1
│   ├── simucode_Study2
│   ├── simucode_Study3
│   ├── simucode_Study4
```

- The data file `para_est_KmeansInsInit.pkl` contains the feature data for the real data based simulation, which is not included in this repository due to its size, but can be obtained at [here](https://stuecnueducn-my.sharepoint.com/:u:/g/personal/10195000464_stu_ecnu_edu_cn/ERJ1kPBzfVZLlRKJ29fUBPUBixmUK_oa3yr4gB34kb3-VQ?e=X1RW57), or by contacting me via email. 
  
We use Study2 as an example to illustrate the file tree as follows:

```
├── simucode_Study2
│   ├── estimation.py
│   ├── plot_prop_Study2.R
│   └── simu.py
```

- `estimation.py` implements the proposed estimating methods.
- `plot_prop_Study2.R` contains the R code for generating figures. 
- `simu.py` simulates data and estimates parameters.

## PART 2. Simulation Studies for Robustness Analysis

The following code files in the `Robustness` folder can be used to reproduce the simulation results for robustness analysis presented in the paper. Settings are similar to those in the `Simulation` folder as before.

```
├── Robustness
│   ├── Study1_hetePi
│   ├── Study2_SpatialCorr
│   ├── Study3_ConIndep
│   ├── para_est_KmeansInsInit.pkl
```

## PART 3. Real Data Analysis on the CAMELYON16 Dataset

The following code files in the `RealDataAnalysis` folder can be used to reproduce real data analysis results presented in the paper. The file tree is given as follows:

```
├── RealDataAnalysis
│   ├── LICENSE
│   ├── estimation.py
│   ├── gentrain.py
│   ├── mul_gentest.py
│   ├── pred.py
│   ├── sin_gentest.py
│   ├── train_est.py
│   └── utils.py
```

The CAMELYON16 dataset can be downloaded at [its official website](https://camelyon16.grand-challenge.org/Data/). Please also note that the model file `pytorch_model.bin` is not involved in this repository due to its size. It can be obtained via email or downloaded at [Hugging Face](https://huggingface.co/MahmoodLab/UNI).

Below are instructions on how to execute the scripts:

### Step 1. Generating the embedded features

We need to generate the features and labels for the training and testing datasets separately.

```
python gentrain.py
python mul_gentest.py
```

### Step 2. Parameter estimation

Next, we apply the proposed estimators to estimate the unknown parameters.

```
python train_est.py
```

### Step 3. Prediction

Finally, the predictive results can be then obtained.

```
python pred.py
```

## Contact

If you have any problem, please feel free to contact [Baichen Yu](mailto:baichen.yu@stu.pku.edu.cn) and [Prof. Xuetong Li](mailto:xtong_li@xjtu.edu.cn).