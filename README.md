# HINT

A generation model equipped with **HI**gh-level representations for lo**N**g **T**ext generation described in the paper [Long Text Generation by Modeling Sentence-Level and Discourse-Level Coherence]() (ACL 2021 Long Paper).



## Prerequisites

The code is written in TensorFlow library. To use the program the following prerequisites need to be installed.

- Python 3.7.0
- torch==1.4.0
- transformers==4.0.0
- pytorch-lightning==1.1.0



## Computing infrastructure

We train HINT based on the platform: 

- OS: Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-98-generic x86_64)
- CUDA Version: 10.1
- GPU: NVIDIA TITAN X (Pascal)



## Quick Start

#### 1. Constructing Training Examples

The full data can be downloaded from [THUcloud](https://cloud.tsinghua.edu.cn/d/e895f635cb4d485d8f98/) or [GoogleDrive](https://drive.google.com/drive/folders/1i_2YfzpDnfuLyyctOyDabn3Br0OcK1Tj?usp=sharing). The structure for the directory `data` is as follows

```markdown
├── data
   └── `pro_data.py`             # the code to create training examples
   └── `preprocess.sh`   # the script to create training examples
   └── `ini_data`		# the directory for the inital data
       ├── `roc`        # ROCStories
              └── `train.txt`        # the full texts including inputs and outputs (sentences separated by [SEP])
              └── `train.source`    # only inputs
              └── `train.target`       # only outputs
              └── ...
       ├── `wp`        # WritingPrompts
              └── ...
       ├── `bc`      # BookCorpus
              └── ...

   └── `data`		# training examples
       ├── `roc`        # ROCStories
              └── `train.source`    # only inputs
              └── `train.target`       # only outputs
              └── `train_order.target`       # file for recording sentence orders
              └── `train_sbertscore.target`       # file for recording the computed sbert score between sentences              
              └── ...
```




#### 2. Post-Training on BookCorpus

Execute the following command (or run `bash ./finetune.sh` directly) to post-train BART on BookCorpus: 

```shell
cd ./model
env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3.7 -u finetune.py \
    --data_dir ../Data/data/bc \
    --output_dir=./bart_post_bc \
    --save_top_k 80 \
    --train_batch_size=10 \
    --eval_batch_size=10 \
    --num_train_epochs 10 \
    --model_name_or_path ./bart_model \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --overwrite_output_dir \
    --sbert \
    --reorder \
```

The initial checkpoint of BART can be downloaded from [BART](https://huggingface.co/facebook/bart-base/tree/main). We use the base version of BART. We train the model for about 0.1M steps. The training process will task about 1~2 days. The post-trained model can be downloaded from [THUcloud](https://cloud.tsinghua.edu.cn/d/eecac06e0d2f479d964d/) or [GoogleDrive](https://drive.google.com/drive/folders/1iBM3UotohMvmeTfkFJJqWS5zmBLOwugb?usp=sharing).



#### 3. Fine-tuning on ROCStories/WritingPrompts

Execute the following command (or run `bash ./finetune.sh` directly) to fine-tune HINT on ROCStories/WritingPromts: 

```shell
cd ./model
env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3.7 -u finetune.py \
    --data_dir ../Data/data/roc \ # ../data/data/wp
    --output_dir=./bart_post_bc_ft_roc \
    --save_top_k 80 \
    --train_batch_size=10 \
    --eval_batch_size=10 \
    --num_train_epochs 10 \
    --model_name_or_path ./bart_post_bc \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --overwrite_output_dir \
```

You can add `--sbert` and `--reorder` to use the proposed two pretraining tasks as the auxiliary tasks for fine-tuning.



#### 4. Generation and Computing Perplexity

Execute the following command to generate texts: 

```shell
cd ./eval
device=cuda:0
model_name_path=../model/bart_post_bc_ft_roc
data_dir=../Data/ini_data/roc
env CUDA_VISIBLE_DEVICES=1 python3.7 ./gen.py $device $model_name_path $data_dir
env CUDA_VISIBLE_DEVICES=1 python3.7 ./ppl.py $device $model_name_path $data_dir
```

The generation results will be saved under the `results` directory.



#### 5. Evaluation

Execute the following command to generate texts: 

```shell
cd ./eval
python3.7 ./eval.py
```

You can change `result_list` in the script to decide the results you want to evaluate.



### Citation

Please kindly cite our paper if [this paper](https://arxiv.org/abs/2009.07602) and the [code](https://github.com/thu-coai/UNION) are helpful.

```
To appear
```