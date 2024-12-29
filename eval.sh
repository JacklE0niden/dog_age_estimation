export PYTHONPATH=$PYTHONPATH:/mnt/pami26/zengyi/dlearning/dog_age_estimation
# python -m torch.distributed.run --nproc_per_node=4 ./train/train.py

tmux new -s eval500
export PYTHONPATH=$PYTHONPATH:/mnt/pami26/zengyi/dlearning/dog_age_estimation
conda activate dog_age_estimation
python -m torch.distributed.run --nproc_per_node=4 eval/evaluate_ddp.py --phase evaluate --data_dir data