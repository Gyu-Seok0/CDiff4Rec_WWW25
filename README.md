# Collaborative Diffusion Model for Recommender System (CDiff4Rec)

### 1. Overview
This repository provides the source code of our paper: Collaborative Diffusion Model for Recommender System (CDiff4Rec).

<img src="./figure/method.png">

### 2. Usage

To run the code, use the run.sh script and specify the dataset name from the options: 'yelp', 'amazon_game', or 'citeulike_t' like below:

```
./run.sh amazon_game
```

### 3. Environment
The code is written in Python 3.9.0 and requires the following dependencies:

* pytorch==1.13.1
* pytorch-cuda=11.6
* numpy=1.24.3

<br> For more details, refer to the environment.yml file.