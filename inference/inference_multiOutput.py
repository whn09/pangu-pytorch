# wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
# sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
# sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cudnn

import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datetime import datetime, timedelta
from tqdm import tqdm
import onnxruntime as ort
import onnx
import numpy as np
from torch.utils import data
from torch import nn
import torch

from era5_data.config import cfg
from era5_data import score
from era5_data import utils, utils_data

from models.pangu_sample import get_wind_speed

visualize = True  # True/False
only_use_wind_speed_loss = True  # True/False
use_custom_mask = True  # True/False
lead_time = 10  # TODO Forecast 10 days

# The directory of your input and output data
PATH = cfg.PG_INPUT_PATH
output_data_dir = cfg.PG_OUT_PATH

# # Load pretrained model
# model_24 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_24)
# # model_6 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_6)
# # model_3 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_3)
# # model_1 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_1)

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = cfg.GLOBAL.NUM_THREADS

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested', }

# providers = [('CUDAExecutionProvider', cuda_provider_options)]
providers = [('CUDAExecutionProvider', {'device_id': 0})]

# A test for a single input frame
# desiered output: future 14 days forecast

h = cfg.PG.HORIZON
output_data_dir = os.path.join(output_data_dir, 'inference', str(h))
utils.mkdirs(output_data_dir)

if h == 24:
    ort_session = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_24, sess_options=options, providers=providers)
elif h == 6:
    ort_session = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_6, sess_options=options, providers=providers)
elif h == 3:
    ort_session = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_3, sess_options=options, providers=providers)
elif h == 1:
    ort_session = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_1, sess_options=options, providers=providers)

# Load mean and std of the weather data
# weather_surface_mean, weather_surface_std = utils.LoadStatic()

# Prepare for the test data
# test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
test_dataset = utils_data.PTDataset(pt_path=PATH,
                                    data_transform=None,
                                    training=False,
                                    validation=False,
                                    startDate=cfg.PG.TEST.START_TIME,
                                    endDate=cfg.PG.TEST.END_TIME,
                                    freq=cfg.PG.TEST.FREQUENCY,
                                    horizon=h,
                                    device='cpu')
dataset_length = len(test_dataset)

test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                  drop_last=True, shuffle=False, num_workers=8, pin_memory=False)  # default: num_workers=0

# Loss function
criterion = nn.L1Loss(reduction='none')

# Load constants and teleconnection indices
aux_constants = utils_data.loadAllConstants(
    device='cpu')  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
upper_weights, surface_weights, upper_loss_weight, surface_loss_weight = aux_constants['variable_weights']

test_loss = 0.0

mask = None
if use_custom_mask:
    mask = aux_constants['custom_mask']
    # 为每个张量添加合适的维度
    mask_3d = mask[None, :, :]  # for 3D tensors
    mask_4d = mask[None, None, :, :]  # for 4D tensors
    valid_points = mask.sum()  # 用于计算平均值

def save_prediction(output, output_surface, hour, prediction_time, save_dir):
    """保存预测结果到磁盘"""
    # 创建以日期命名的子目录
    date_dir = os.path.join(save_dir, prediction_time.strftime('%Y%m%d'))
    os.makedirs(date_dir, exist_ok=True)
    
    # 保存文件名格式：YYYYMMDD_HH.npz
    save_path = os.path.join(date_dir, f"{prediction_time.strftime('%Y%m%d_%H')}.npz")
    
    # 保存为压缩的npz文件
    np.savez_compressed(
        save_path,
        output=output,
        output_surface=output_surface,
        hour=hour,
        timestamp=prediction_time.strftime('%Y%m%d%H')
    )
    
    return save_path

batch_id = 0
# 每天00:00 12:00预报h小时候的天气（single frame output）
# for id, data in tqdm(enumerate(test_dataloader, 0)):
for data in tqdm(test_dataloader):
    # Dic to save rmses
    rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_upper_wind_speed = dict(
    ), dict(), dict(), dict(), dict(), dict()
    rmse_surface = dict()
    rmse_surface_wind_speed = dict()

    acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(
    ), dict(), dict(), dict(), dict()
    acc_surface = dict()

    # Store initial input for different models
    input, input_surface, target, target_surface, periods = data
    print('periods:', periods)
    # print('input:', input.shape)
    # print('input_surface:', input_surface.shape)
    # print('target:', target.shape)
    # print('target_surface:', target_surface.shape)
    
    # print('periods[0][batch_id]:', periods[0][batch_id])
    if not periods[0][batch_id].endswith('00'):  # 只评估从0点开始的结果
        continue

    # Required input to the pretrained model: upper ndarray(n, Z, W, H) and surface(n, W, H)
    input, input_surface = input.numpy().astype(np.float32).squeeze(), input_surface.numpy(
    ).astype(np.float32).squeeze()  # input torch.Size([1, 5, 13, 721, 1440])

    # spaces = h // 24  # TODO: may change
    freq = int(cfg.PG.TEST.FREQUENCY[:-1])
    # spaces = h // freq
    spaces = lead_time * 24 // freq
    # start time
    input_time_str = periods[0][batch_id]
    input_time = datetime.strptime(input_time_str, '%Y%m%d%H')
    save_dir = os.path.join(output_data_dir, 'predictions', input_time_str)

    # multi-step prediction for single output
    for space in range(spaces):
        current_time = input_time + timedelta(hours=freq*(space+1))
        print("predicting on....", current_time)

        # Call the model pretrained for 24 hours forecast
        start = time.time()
        output, output_surface = ort_session.run(
            None, {'input': input, 'input_surface': input_surface})
        end = time.time()
        # print('ort_session.run time:', end-start)

        # save_path = save_prediction(output, output_surface, h, current_time, save_dir)
        # end2 = time.time()
        # print('save_prediction time:', end2-end)
        
        # Stored the output for next round forecast
        input, input_surface = output.copy(), output_surface.copy()
        
        if space>0:
            _, _, target, target_surface, periods = test_dataset[test_dataset.keys.index(current_time)-1]  # TODO
            target = target.unsqueeze(0)
            target_surface = target_surface.unsqueeze(0)
            periods = [[periods[0]], [periods[1]]]
            # print('periods:', periods)

        # make sure the predicted time step is the same as the target time step
        assert current_time == datetime.strptime(periods[1][batch_id], '%Y%m%d%H')
        target_time = periods[1][batch_id]

        output, output_surface = torch.from_numpy(output).type(
            torch.float32), torch.from_numpy(output_surface).type(torch.float32)
        
        output_surface_wind_speed, target_surface_wind_speed, output_wind_speed, target_wind_speed = get_wind_speed(output_surface.unsqueeze(0), target_surface, output.unsqueeze(0), target)
        
        # Noralize the gt to make the loss compariable
        target_normalized, target_surface_normalized = utils_data.normData(target, target_surface, aux_constants['weather_statistics_last'])
        output_normalized, output_surface_normalized = utils_data.normData(output, output_surface, aux_constants['weather_statistics_last'])

        if only_use_wind_speed_loss:
            output_surface_wind_speed_normalized, target_surface_wind_speed_normalized, output_wind_speed_normalized, target_wind_speed_normalized = get_wind_speed(output_surface_normalized, target_surface_normalized, output_normalized, target_normalized)
            if use_custom_mask:
                surface_wind_speed_loss = (criterion(output_surface_wind_speed_normalized, target_surface_wind_speed_normalized) * mask_3d).sum() / valid_points
                wind_speed_loss = (criterion(output_wind_speed_normalized, target_wind_speed_normalized) * mask_3d).sum() / valid_points
                loss = surface_wind_speed_loss + wind_speed_loss
            else:
                surface_wind_speed_loss = criterion(output_surface_wind_speed_normalized, target_surface_wind_speed_normalized)
                wind_speed_loss = criterion(output_wind_speed_normalized, target_wind_speed_normalized)
                loss = torch.mean(surface_wind_speed_loss) + torch.mean(wind_speed_loss)
        else:
            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss_surface = criterion(output_surface_normalized, target_surface_normalized)
            loss_upper = criterion(output_normalized, target_normalized)
            
            if use_custom_mask:
                # 应用mask并计算有效区域的平均损失
                weighted_surface_loss = (loss_surface * surface_weights * mask_4d).sum() / (valid_points * loss_surface.size(1))
                weighted_upper_loss = (loss_upper * upper_weights * mask_3d).sum() / valid_points
            else:
                weighted_surface_loss = torch.mean(loss_surface * surface_weights)
                weighted_upper_loss = torch.mean(loss_upper * upper_weights)
                
            # The weight of surface loss is 0.25
            # loss = weighted_upper_loss + weighted_surface_loss * 0.25
            loss = weighted_upper_loss * upper_loss_weight + weighted_surface_loss * surface_loss_weight  # change loss weight
            
        test_loss += loss.item()
        
        target, target_surface = target.squeeze(), target_surface.squeeze()
        output, output_surface = output.squeeze(), output_surface.squeeze()
        
        # output_surface_wind_speed = output_surface_wind_speed.squeeze()
        # target_surface_wind_speed = target_surface_wind_speed.squeeze()
        output_wind_speed = output_wind_speed.squeeze()
        target_wind_speed = target_wind_speed.squeeze()
        
        # print('target_time:', target_time)
        # print(output_surface_wind_speed.shape, target_surface_wind_speed.shape, output_wind_speed.shape, target_wind_speed.shape)
        # print('output_surface_wind_speed:', output_surface_wind_speed)
        # print('target_surface_wind_speed:', target_surface_wind_speed)
        # print('output_wind_speed:', output_wind_speed)
        # print('target_wind_speed:', target_wind_speed)
        
        # mslp, u,v,t2m 3: visualize t2m
        if visualize:
            png_path = os.path.join(output_data_dir, "png")
            if not os.path.exists(png_path):
                os.mkdir(png_path)
                
            utils.visuailze(output,
                            target, 
                            input.astype(np.float32).squeeze(),  # .numpy()
                            var='t',
                            z=2,
                            step=target_time, 
                            path=png_path)

            utils.visuailze_surface(output_surface,
                                    target_surface, 
                                    input_surface.astype(np.float32).squeeze(),  # .numpy()
                                    var='u10',
                                    step=target_time, 
                                    path=png_path)
            
            utils.visuailze_surface(output_surface,
                                    target_surface, 
                                    input_surface.astype(np.float32).squeeze(),  # .numpy()
                                    var='v10',
                                    step=target_time, 
                                    path=png_path)
            
        rmse_upper_z[target_time] = score.weighted_rmse_torch_channels(
            output[0], target[0], mask).numpy()
        rmse_upper_q[target_time] = score.weighted_rmse_torch_channels(
            output[1], target[1], mask).numpy()
        rmse_upper_t[target_time] = score.weighted_rmse_torch_channels(
            output[2], target[2], mask).numpy()
        rmse_upper_u[target_time] = score.weighted_rmse_torch_channels(
            output[3], target[3], mask).numpy()
        rmse_upper_v[target_time] = score.weighted_rmse_torch_channels(
            output[4], target[4], mask).numpy()
        rmse_upper_wind_speed[target_time] = score.weighted_rmse_torch_channels(
            output_wind_speed, target_wind_speed, mask).numpy()
        rmse_surface[target_time] = score.weighted_rmse_torch_channels(
            output_surface, target_surface, mask).numpy()
        rmse_surface_wind_speed[target_time] = score.weighted_rmse_torch_channels(
            output_surface_wind_speed, target_surface_wind_speed, mask).numpy()

        # acc TODO: need to support mask
        surface_mean, _, upper_mean, _ = utils_data.weatherStatistics_output(filepath=os.path.join(cfg.PG_INPUT_PATH, 'aux_data'), device='cpu')
        output_anomaly = output - upper_mean.squeeze(0)
        target_anomaly = target - upper_mean.squeeze(0)

        output_surface_anomaly = output_surface - surface_mean.squeeze(0)
        target_surface_anomaly = target_surface - surface_mean.squeeze(0)
        acc_upper_z[target_time] = score.weighted_acc_torch_channels(
            output_anomaly[0], target_anomaly[0]).numpy()
        acc_upper_q[target_time] = score.weighted_acc_torch_channels(
            output_anomaly[1], target_anomaly[1]).numpy()
        acc_upper_t[target_time] = score.weighted_acc_torch_channels(
            output_anomaly[2], target_anomaly[2]).numpy()
        acc_upper_u[target_time] = score.weighted_acc_torch_channels(
            output_anomaly[3], target_anomaly[3]).numpy()
        acc_upper_v[target_time] = score.weighted_acc_torch_channels(
            output_anomaly[4], target_anomaly[4]).numpy()

        acc_surface[target_time] = score.weighted_acc_torch_channels(output_surface_anomaly,
                                                                    target_surface_anomaly).numpy()

    # Save rmse,acc to csv
    csv_path = os.path.join(output_data_dir, input_time_str, "csv")
    utils.mkdirs(csv_path)

    utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_upper_wind_speed, rmse_surface, rmse_surface_wind_speed,
                        "rmse")
    utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q,
                        acc_upper_t, acc_upper_u, acc_upper_v, None, acc_surface, None, "acc")
    
    break

test_loss /= len(test_dataloader)
print('test_loss:', test_loss)