export PYTHONPATH=$PYTHONPATH:/mnt/pami26/zengyi/dlearning/dog_age_estimation
# python -m torch.distributed.run --nproc_per_node=4 ./train/train.py
# python ./train/train.py
python -m torch.distributed.run --nproc_per_node=6 ./train/train_ddp.py
# torchrun --nproc_per_node=4 train_ddp.py

# nohup python -m torch.distributed.run --nproc_per_node=6 ./train/train_ddp.py &> training_output.log &