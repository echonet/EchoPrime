# EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation

This repository contains the official inference code for the following paper:

**EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation**  
*Milos Vukadinovic, Xiu Tang, Neal Yuan, Paul Cheng, Debiao Li, Susan Cheng, Bryan He\*, David Ouyang\**  
[Read the paper on arXiv](https://arxiv.org/abs/2410.09704), 
[See the demo](https://x.com/i/status/1846321746900558097)

![EchoPrime Demo](demo_image.png)

## How To Use
1) Clone the repository and navigate to the EchoPrime directory
2) Download model data 
    * `wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip`
    * `wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p1.pt`
    * `wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p2.pt`
    * `unzip model_data.zip`
    *  `mv candidate_embeddings_p1.pt model_data/candidates_data/`
    *  `mv candidate_embeddings_p2.pt model_data/candidates_data/`
4) Install `requirements.txt`
5) Follow EchoPrimeDemo.ipynb notebook

## Licence
This project is licensed under the terms of the MIT license.


## FAQ:

### After processing the images they appearg green-tinted.
Make sure that you have the correct libraries installed. Use requirements.txt to install the dependencies.


### How to use the view classification model only?
If you are only interested in the view classification task, take a look at the `ViewClassification.ipynb` notebook.

### How to use EchoPrime to predict additional conditions (such as severity-based dilation, regurgitation...)?
If you are only interested in using EchoPrime to predict additional conditions, take a look at the `ExtendedPrediction.ipynb` notebook.

## How to run the code in docker?

```
docker build -t echo-prime .
```

```
docker run -d --name echoprime-container --gpus all echo-prime tail -f /dev/null
```
Then you can attach to this container and run the notebook located at 
`/workspace/EchoPrime/EchoPrimeDemo.ipynb`.