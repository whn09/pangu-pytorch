# conda create -n pangu python=3.10
# conda activate pangu
# pip install -r requirements.txt

# ./download_era5.sh
# python convert_era5.py
# python models/onnx2torch.py

# pip install s5cmd
# s5cmd cp s3://datalab/nsf-ncar-era5/aux_data/* /opt/dlami/nvme/aux_data/
# s5cmd cp s3://datalab/nsf-ncar-era5/pretrained_model/* /opt/dlami/nvme/pretrained_model/

# For horizon=24
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*00.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*00.pt /opt/dlami/nvme/upper/

# For horizon=6
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*00.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*00.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*06.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*06.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*12.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*12.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*18.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*18.pt /opt/dlami/nvme/upper/

# For horizon=3
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*00.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*00.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*03.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*03.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*06.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*06.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*09.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*09.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*12.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*12.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*15.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*15.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*18.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*18.pt /opt/dlami/nvme/upper/
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*21.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*21.pt /opt/dlami/nvme/upper/

# For horizon=1
# s5cmd cp s3://datalab/nsf-ncar-era5/surface/surface_*.pt /opt/dlami/nvme/surface/
# s5cmd cp s3://datalab/nsf-ncar-era5/upper/upper_*.pt /opt/dlami/nvme/upper/

# torchrun --nproc_per_node 1 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 1 --nnodes 1 finetune/lora_tune.py

# torchrun --nproc_per_node 8 --nnodes 1 finetune/finetune_fully.py
# torchrun --nproc_per_node 8 --nnodes 1 finetune/lora_tune.py

# deepspeed --num_gpus=8 models/pangu_model_deepspeed.py 
# deepspeed --num_gpus=8 finetune/finetune_fully.py --use_deepspeed True --only_use_wind_speed_loss True --use_custom_mask True

#  --num_workers 4
#  --num_workers 8
#  --load_pretrained True
#  --load_my_best False
#  --visualize True
#  --only_test True
#  --only_use_wind_speed_loss True
#  --use_custom_mask True
#  --use_deepspeed True


torchrun --nproc_per_node 8 --nnodes 1 finetune/finetune_fully.py --load_pretrained False --load_my_best False --only_use_wind_speed_loss True --use_custom_mask True


# debug
# sudo apt install sysstat
# iostat 2
# tmux new -s session_name
# tmux ls
# tmux detach -s session_name
# tmux attach -t session_name
# s5cmd sync /opt/dlami/nvme/model/ s3://datalab/nsf-ncar-era5/model/
# s5cmd sync s3://datalab/nsf-ncar-era5/model/* /opt/dlami/nvme/model/

# For PanguWeather onnx model
# python inference/inference_multiOutput.py

# For PanguWeather pytorch fine-tuned model
# python inference/inference_mix_multiOutput.py
