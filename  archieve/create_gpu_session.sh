
# 申请一个资源分配（job allocation）包括节点/CPU/内存/GPU 等，并保持这个 allocation 直到退出。
# salloc -p gpu_a100 -t 00:30:00 --cpus-per-task=18 --mem=120G --gpus=1
# srun --pty bash -i

# 或者 直接 
# 在已经分配的资源上启动任务(job step) 而如果当前没有 allocation，那么它也会 隐式创建 allocation 并运行程序。
srun -p gpu_a100 -t 00:30:00 --cpus-per-task=18 --mem=120G --gpus=1 --nodes=1 --ntasks=1 --pty bash -i
#srun -p gpu_h100 -t 00:30:00 --cpus-per-task=18 --mem=120G --gpus=1 --nodes=1 --ntasks=1 --pty bash -i

# 或者 非交互式 sbatch 提交脚本
# sbatch run_timevlm.slurm
# 
# run_timevlm.slurm 内容示例：
    # #!/usr/bin/env bash
    # #SBATCH -p gpu_a100              # 分区
    # #SBATCH --gpus=1                # 1 张 GPU
    # #SBATCH --cpus-per-task=18
    # #SBATCH --mem=120G
    # #SBATCH -t 03:00:00              # walltime（可改 100:00:00）
    # #SBATCH -J TimeVLM_Traffic       # 作业名
    # #SBATCH -o logs/%x_%j.out        # stdout 日志
    # #SBATCH -e logs/%x_%j.err        # stderr 日志

    # # ===== 下面开始是“你平时在 GPU 节点里手敲的命令” =====

    # source ./load_exp_env.sh
    # sh TimeVLM_long_1.0p.sh
