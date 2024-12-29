export PYTHONPATH=$PYTHONPATH:/mnt/pami26/zengyi/dlearning/dog_age_estimation
# python -m torch.distributed.run --nproc_per_node=4 ./train/train.py
# python ./train/train.py
export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.run --nproc_per_node=2 ./train/train_ddp_ss.py
# torchrun --nproc_per_node=4 train_ddp.py

# nohup python -m torch.distributed.run --nproc_per_node=6 ./train/train_ddp.py &> training_output.log &

# 创建新的tmux会话，命名为 'train'
tmux new -s train
conda activate dog_age_estimation
# 在tmux中运行训练命令（按下回车后执行）
python -m torch.distributed.run --nproc_per_node=4 --master_port=50000 ./train/train_ddp.py
python -m torch.distributed.run --nproc_per_node=4 --master_port=50000 ./train/train_ddp_ss.py
# 从tmux会话中分离（按键序列）：
# Ctrl+b 然后按 d

# 列出所有tmux会话
tmux ls

# 重新连接到train会话
tmux attach -t train


# 结束会话
tmux kill-session -t train

# 创建新窗口: Ctrl+b 然后按 c
# 切换窗口: Ctrl+b 然后按 窗口编号(0,1,2...)
# 分割窗口为上下两个面板: Ctrl+b 然后按 "
# 分割窗口为左右两个面板: Ctrl+b 然后按 %
# 在面板之间切换: Ctrl+b 然后按方向键

python -m torch.distributed.run --nproc_per_node=4 ./train/train_ddp.py > training.log 2>&1