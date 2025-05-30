# UFold-X
UFold-X: An Enhanced Dual &amp; Dynamic U-Mamba Model for Long-Range RNA Secondary Structure Prediction

<img src='https://github.com/AI4Bread/UFold-X/blob/main/UFold-X/UFold-X%20icon.png' width=300 height=250>

## Web server
We have developed a user-friendly web server that supports batch submissions and long-sequence prediction. The server is freely accessible at [UFold-X webserver](http://116.211.143.53:23892/UFold-X).

## Introduction
RNA secondary structure is essential for understanding the functional roles of non-coding RNAs, ribosomal RNAs, and viral genomes. However, accurately predicting secondary structures—particularly for long RNA sequences—remains a major challenge due to the complexity of long-range base-pairing interactions and the limited generalization of existing models trained predominantly on short sequences. In this work, we propose UFold-X, a dual-branch U-Net-based deep learning framework specifically designed for long-range RNA secondary structure prediction. UFold-X integrates a fully convolutional network for local feature extraction and a Mamba-based Visual State Space Module (VSSM) for efficient global dependency modeling. A dynamic gating mechanism adaptively fuses outputs from both branches based on sequence length, enabling robust generalization across varying sequence scales. Comprehensive experiments on benchmark datasets demonstrate that UFold-X achieves state-of-the-art performance, significantly outperforming both traditional thermodynamic models and deep learning baselines, particularly in long-sequence and long-range base-pair prediction tasks. Notably, on the RNAstralign-1800 dataset and the unseen ultra-long RCSB4000 dataset, UFold-X improves F1-score by up to 158\% while maintaining fast inference with an average runtime of 0.08 seconds per sequence, showcasing strong generalization to novel RNA families.

## Prerequisites

## Installation 
Clone the repository.

```
git clone https://github.com/AI4Bread/UFold-X.git
```

Navigate to the root of this repo and setup the conda environment.

### !!!Noted: 

### 1. Please change prefix path into your own path in the last line of UFold-X.yml file.

### 2. Please match your python version, cuda version and torch version with the package.

```
conda env create -f UFold-X.yml
```

Activate conda environment.

```
conda activate UFold-X
```

## Pre-trained models 
Pre-trained models are deposited in our [drive](). Please download them and put them into models folder.

## Usage

### Recommended
We recommend users use our [UFold-X webserver](http://116.211.143.53:23892/UFold-X), which is user-friendly and easy to use. Everyone could upload or typein your own candidate RNA sequence in our web without further installation, our backend server will calculate and give the prediction result to the user. User can choose to download the predict ct file result as well as visualize them online directly. We provide the following **three** types of pretrained models:

1. A model designed for predicting RNA sequences **ranging from 600 bp to 1800 bp in length** (highly recommended for long sequences). This model is pretrained on the **RNAstralign-1800 dataset**.

2. A model capable of predicting RNA sequences **ranging from 0 bp to 1800 bp** (suitable for **both short and long sequences**). This model is pretrained on the **RNAstralign-all dataset**.

3. A model which can predict RNA sequences **ranging from 1800 bp to 5000 bp**, although only ~1800 bp yield valid predictions. This model is pretrained on the **RNAstralign-1800 dataset**.

### Try UFold-X

If you want to try UFold-X on your own server, please run:
```python
python ufold_predict.py
```
You can also add optional parameters **-nc True** which indicates support for predicting non-canonical pairs, and default is **False**.

Make sure that the sequence you want to predict has been placed in the **input.txt file under the data folder** in the correct format, just like this:
```
>/Bacteria/B06883.ct
UCCCGGUGAUUGGAGCGCUGUGGCACCACUCCUUCCCAUUCCGAACAGGAUAGUGAAAGGCAGCAGCGGGUACGAUACUUGAAUCGCAAGGAUCUGGGACAAUACCUCAUCGCCGGGU
>./RNAStrAlign/tRNA_database/tdbD00004755.ct
GCCCCCAUCGUCUAACGGUUAGGACACCAGACUUUCAAUCUGACAACGAGAGUUCGACUCUCUCUGGGGGUA
```
After running the above command, you will get the **output ct file, bpseq file, and figure** in the results folder.

### Evaluate UFold-X

### Train models based on your own datasets

## Contribute
We love your input! We want to make contributing to UFold as easy and transparent as possible. Please see our [Contributing Guide]() to get started. Thank you to all our contributors!

## Citation
If you use our tool, please cite our work: 

UFold-X: An Enhanced Dual & Dynamic U-Mamba Model for Long-Range RNA Secondary Structure Prediction

