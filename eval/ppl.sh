cd ./eval
device=cuda:0
model_name_path=../model/bart_post_bc_ft_roc
data_dir=../Data/ini_data/roc
env CUDA_VISIBLE_DEVICES=1 python3.7 ./ppl.py $device $model_name_path $data_dir