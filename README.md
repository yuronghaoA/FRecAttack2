This project contains the source codes of the proposed approach and experiments for the paper *Not One Less: Exploring Interplay between User Profiles and Items in Untargeted Attacks against Federated Recommendation*, accepted by ACM CCS 2024. In the following, we will explain the prerequisites to run the codes and the file structure of the project.

# Preparation

## Virtual Environment Creation

The project is coded with Python 3.8 and PyTorch 1.12.0. The following shell commands are used to create the virtual environment with **Anaconda**.

```shell
conda create -n FR python\=3.8
conda activate FR
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio\=\=0.12.0 cudatoolkit\=11.6 -c pytorch 
[it may take 3 to 5 minutes.]
```

## Package and Library Installation

The following commands are used to install the required packages and libraries in the virtual environment **FR** to accelerate the speed of data processing and analysis.

```shell
pip install -r requirements.txt        [approximately 25 minutes to install]
conda install -c <https://conda.anaconda.org/rapidsai> -c numba -c <https://conda.anaconda.org/nvidia> -c conda-forge cudf\=23.04 cuml\=23.04      [approximately 40 minutes to install]
```

# Getting Started

With the prerequisites for running this project in place, we will present an overview of the project's file structure.  An example and specific steps will be provided to assist in running the project.

## File Tree

*   **Data**: contains datasets used in the experiment

*   **dataloader.py**: responsible for loading data

*   **main.py**: the main script

*   **server.py**: the operations performed by the aggregator

*   **client.py**: the local operations performed by benign users

*   **attack.py**: the local operations performed by malicious users

*   **col\_server.py**: the operations performed by malicious users when they collude

*   **defense.py**: the operations performed by the aggregator when deploying GuardCQ defence

*   **model.py**: the base recommender models, containing FedNCF, FedMLP, FedSoG, FedGNN

*   **GAT.py**: Sub-modules of FedSoG and FedGNN

*   **filename\_gen.py**: used to generate filenames for experiment results according to the command line in `run.sh`

*   **parse.py**: lists the parameters used in the codes

*   **utils.py**: implementation of the experimental metrics, e.g., HR and NDCG

*   **run.sh**: shell codes required to run the project

*   **results\_analysis.py**: used to generate the experimental results

## Example:

Here's an example command to run the attack with Col-SH on the *MovieLens* (ML-1M) dataset with FedNCF:

     python main.py --select_model FedNCF --data ML_1M --lr 0.005 --epoch 2500 --mali_ratio 0.1 --attack_user ColSH --attack_item RatingOfChange --sample_size 100 --alpha 0.1 --agg avg --show_mode print & 

The output similar to the following will be shown in the terminal:

    Arguments: show_mode=print,select_model=FedNCF,data=ML_1M,device=cuda,layers=[64, 32, 16, 8],batch_size=16,embedding_dim=16,lr=0.005,epoch=2500,frac=0.1,num_neg=1,seed=608,top_k=[10,20],agg=avg,valid_step=1,mali_ratio=0.1,attack_user=ColSH,Noisy_pat=ColDP,attack_item=RatingOfChange,sample_size=100,alpha=0.1,sigma=0.1,window_size=2,is_detect=1,start_detect=1,wind_sz=50,clip=0.1,laplace_lambda=0.1,loss=mae,weight_decay=0.001,head_num=1 
    Using backend: pytorch
    Iteration 0, loss = 0.68924, HR@10 = 0.00258, nDCG@10 = 0.00102 
    Iteration 1, loss = 0.69012, HR@10 = 0.00276, nDCG@10 = 0.00108 
    Iteration 2, loss = 0.68959, HR@10 = 0.00331, nDCG@10 = 0.00122 
    Iteration 3, loss = 0.68903, HR@10 = 0.00368, nDCG@10 = 0.00140 
    Iteration 4, loss = 0.68851, HR@10 = 0.00350, nDCG@10 = 0.00140 
    Iteration 5, loss = 0.68784, HR@10 = 0.00331, nDCG@10 = 0.00140 
    Iteration 6, loss = 0.68727, HR@10 = 0.00350, nDCG@10 = 0.00150 
    Iteration 7, loss = 0.68675, HR@10 = 0.00386, nDCG@10 = 0.00163 
    Iteration 8, loss = 0.68608, HR@10 = 0.00405, nDCG@10 = 0.00164 
    Iteration 9, loss = 0.68585, HR@10 = 0.00386, nDCG@10 = 0.00160 
    Iteration 10, loss = 0.68529, HR@10 = 0.00460, nDCG@10 = 0.00182 
    Iteration 11, loss = 0.68491, HR@10 = 0.00478, nDCG@10 = 0.00192 
    Iteration 12, loss = 0.68466, HR@10 = 0.00497, nDCG@10 = 0.00202 
    Iteration 13, loss = 0.68459, HR@10 = 0.00533, nDCG@10 = 0.00223 
    Iteration 14, loss = 0.68437, HR@10 = 0.00533, nDCG@10 = 0.00220 
    ... 

## Run Experiment

If you wish to obtain comprehensive results and analysis for our attacks and defense, you can refer to command `CUDA_VISIBLE_DEVICES=0 nohup bash run_attack.sh &` and `CUDA_VISIBLE_DEVICES=0 nohup bash run_defense.sh &`.

*   If there are multiple GPUs available, you can modify "CUDA\_VISIBLE\_DEVICES\=0" (perhaps CUDA\_VISIBLE\_DEVICES\=1,2 or 3) to reduce the running time.

*   Once started, it will automatically create a directory called `results` and other sub-directories according to the command line.

*   The files storing the results are named in a unified format:

> {attack method}-{the ratio of malicious user}-{defense method}-{random seed}.txt

Running  `results_analysis.py` will provide you with the main results related to both the attack and defense.&#x20;

\
*Please note that the optimal hyperparameters may vary across different datasets or models.*



# Citation

Consider citing the paper if you use the codes in your papers, as follows:

    @article{NotOneLess,
      title={Not One Less: Exploring Interplay between User Profiles and Items in Untargeted Attacks against Federated Recommendation.},
      author={Yurong Hao and Xihui Chen and Xiaoting Lyu and Jiqiang Liu and Yongshen Zhu and Zhiguo Wan and Sjouke Mauw and Wei Wang},
      journal={ACM Conference on Computer and Communications Security (CCS)},
      year={2024}
    }

# License

The project is for learning and research purposes only.
