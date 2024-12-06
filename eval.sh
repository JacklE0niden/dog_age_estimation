export PYTHONPATH=$PYTHONPATH:/mnt/pami26/zengyi/dlearning/dog_age_estimation
# python -m torch.distributed.run --nproc_per_node=4 ./train/train.py
python -m torch.distributed.run --nproc_per_node=6 eval/evaluate_ddp.py --phase evaluate --data_dir data