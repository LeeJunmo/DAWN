# [WSDM'25] Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www.wsdm-conference.org/2025/" alt="Conference">
        <img src="https://img.shields.io/badge/WSDM'25-brightgreen" /></a>
</p>

The source code of the novel method for **D**etecting fake news via e**A**rliness-guided re**W**eighti**N**g ( **DAWN** ) from the paper "[Revisiting Fake News Detection: Towards Temporality-aware Evaluation by Leveraging Engagement Earliness]()" at WSDM 2025.

Junghoon Kim, Junmo Lee, Yeonjun In, Kanghoon Yoon, and Chanyoung Park

## Abstract
Social graph-based fake news detection aims to identify news articles containing false information by utilizing social contexts, e.g., user information, tweets and comments. However, conventional methods are evaluated under less realistic scenarios, where the model has access to future knowledge on article-related and context-related data during training. In this work, we newly formalize a more realistic evaluation scheme that mimics real-world scenarios, where the data is \textit{temporality-aware} and the detection model can only be trained on data collected up to a certain point in time. We show that the discriminative capabilities of conventional methods decrease sharply under this new setting, and further propose~\proposed, a method more applicable to such scenarios. Our empirical findings indicate that later engagements (e.g., consuming or reposting news) contribute more to noisy edges that link real news-fake news pairs in the social graph. Motivated by this, we utilize feature representations of engagement earliness to guide an edge weight estimator to suppress the weights of such noisy edges, thereby enhancing the detection performance of \proposed. Through extensive experiments, we demonstrate that \proposed~outperforms existing fake news detection methods under real-world environments.

## Overall Architecture
<p align="center"><img width="1000" src="./images/architecture.png"></p>

## Requirements
- python=3.9.18
- pytorch=1.12.1
- scikit-learn=1.3.0
- numpy=1.26.3
- scipy=1.11.4 
- pyg=2.4.0 (torch-geometric)
- twitter=1.19.6

## How to Run?
You can run the model as following commands.

- python file for training/testing DAWN
```bash
run.py
```
- shell file for running ```run.py``` with our best performance hyperparameter set.
```bash
sh run.sh
```

## Data Download/generation

You can download two datasets in this drive [link](https://drive.google.com/drive/folders/1YbHgef66Jkf_0EaT7GTWm8VmEb3eSEi7?usp=sharing)

Download each folder and place it in the current directory. (We upload our datasets into drive link due to large file size.)

Also you can generate a customized dataset with different hyperparameter settings (e.g. deadline, threshold)
```bash
construct_data.py
```
