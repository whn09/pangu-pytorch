python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 finetune/finetune_fully.py
# python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 finetune/lora_tune.py