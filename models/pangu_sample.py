import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tqdm import tqdm
import copy
import time
import subprocess

import torch
from torch import nn

from era5_data import score
from era5_data.config import cfg
from era5_data import utils, utils_data


def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        return [line.split(', ') for line in output.decode('utf-8').strip().split('\n')]
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None

    
def get_disk_info():
    try:
        output = subprocess.check_output(['df', '-h'], universal_newlines=True)
        lines = output.strip().split('\n')
        return [line.split() for line in lines[1:]]  # Skip the header
    except subprocess.CalledProcessError:
        return None

    
def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0

        
def monitor_system(interval=5, duration=60):
    end_time = time.time() + duration
    while time.time() < end_time:
        print("\n" + "="*50)
        print(f"System Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

        # Monitor GPU
        gpu_info = get_gpu_info()
        if gpu_info:
            print("\nGPU Status:")
            for i, (util, mem_used, mem_total) in enumerate(gpu_info):
                print(f"GPU {i}: Utilization {util}%, Memory {mem_used}/{mem_total} MB")
        else:
            print("\nGPU information not available")

        # Monitor Disk
        disk_info = get_disk_info()
        if disk_info:
            print("\nDisk Usage:")
            for filesystem, size, used, avail, use_percent, mounted_on in disk_info:
                print(f"{mounted_on}: {use_percent} used ({used}/{size})")
        else:
            print("\nDisk information not available")

        time.sleep(interval)


def train(model, train_loader, val_loader, optimizer, lr_scheduler, res_path, device, writer, logger, start_epoch,
          rank=0, visualize=False):
    '''Training code'''
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    criterion = nn.L1Loss(reduction='none')

    # training epoch
    epochs = cfg.PG.TRAIN.EPOCHS
    accumulation_steps = cfg.PG.TRAIN.ACCUMULATION_STEPS

    loss_list = []
    best_loss = float('inf')
    epochs_since_last_improvement = 0
    best_model = None
    # scaler = torch.cuda.amp.GradScaler()

    # Load constants and teleconnection indices
    aux_constants = utils_data.loadAllConstants(
        device=device)  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
    upper_weights, surface_weights = aux_constants['variable_weights']

    # Train a single Pangu-Weather model
    for i in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # for id, train_data in enumerate(train_loader):
        for iter_num, train_data in enumerate(tqdm(train_loader, desc=f'Training epoch {i} rank {rank}')):
            # if rank == 0:
            #     monitor_system(interval=1, duration=1)
                # print(f'Epoch: {i}', '*'*20)
                # for root, dirs, files in os.walk(cfg.PG_INPUT_PATH):
                #     for filename in files:
                #         filepath = os.path.join(root, filename)
                #         size = os.path.getsize(filepath)
                #         print(f'Epoch: {i}', os.path.join(root, filename), size/1024)
                # print(f'Epoch: {i}', '#'*20)

            if (iter_num + 1) % accumulation_steps == 0:
                optimizer.zero_grad()

            # Load weather data at time t as the input; load weather data at time t+336 as the output
            # Note the data need to be randomly shuffled
            input, input_surface, target, target_surface, periods = train_data
            # print('input:', input.shape, 'input_surface:', input_surface.shape, 'target:', target.shape, 'target_surface:', target_surface.shape, 'periods:', periods)
            input, input_surface, target, target_surface = input.to(device), input_surface.to(device), target.to(
                device), target_surface.to(device)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # /with torch.cuda.amp.autocast():

            # Note the input and target need to be normalized (done within the function)
            # Call the model and get the output
            output, output_surface = model(input, input_surface, aux_constants['weather_statistics'],
                                           aux_constants['constant_maps'],
                                           aux_constants['const_h'])  # (1,5,13,721,1440)

            # Normalize gt to make loss compariable
            target, target_surface = utils_data.normData(
                target, target_surface, aux_constants['weather_statistics_last'])
            
            # print(f"output.shape: {output.shape}, output_surface.shape: {output_surface.shape}")

            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss_surface = criterion(output_surface, target_surface)
            weighted_surface_loss = torch.mean(loss_surface * surface_weights)

            loss_upper = criterion(output, target)
            weighted_upper_loss = torch.mean(loss_upper * upper_weights)
            # The weight of surface loss is 0.25
            loss = weighted_upper_loss + weighted_surface_loss * 0.25

            # Call the backward algorithm and calculate the gratitude of parameters
            # scaler.scale(loss).backward()
            loss = loss / accumulation_steps  # 将损失除以累积步数
            loss.backward()

            # Update model parameters with Adam optimizer
            # scaler.step(optimizer)
            # scaler.update()
            if (iter_num + 1) % accumulation_steps == 0:
                optimizer.step()
                
            epoch_loss += loss.item()
                
        epoch_loss /= len(train_loader)
        epoch_end = time.time()
        logger.info("Epoch {} Rank {}: loss={:.3f}, time={:.3f}".format(i, rank, epoch_loss, epoch_end-epoch_start))
        
        loss_list.append(epoch_loss)
        lr_scheduler.step()
        # scaler.update(lr_scheduler)
        #
        # for name, param in model.named_parameters():
        #   writer.add_histogram(name, param.data, i)
        
        # Clean up memory
        del input, input_surface, target, target_surface, output, output_surface
        torch.cuda.empty_cache()

        if rank == 0:
            model_save_path = os.path.join(res_path, 'models')
            utils.mkdirs(model_save_path)

            # Save the training model
            if i % cfg.PG.TRAIN.SAVE_INTERVAL == 0:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": i}
                torch.save(save_file, os.path.join(
                    model_save_path, 'train_{}.pth'.format(i)))
                # torch.save(model, os.path.join(model_save_path,'train_{}.pth'.format(i)))

            # Begin to validate
            if i % cfg.PG.VAL.INTERVAL == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0.0
                    # for id, val_data in enumerate(val_loader, 0):
                    for val_data in tqdm(val_loader, desc='Validating'):
                        input_val, input_surface_val, target_val, target_surface_val, periods_val = val_data
                        # input_val_raw, input_surface_val_raw = input_val, input_surface_val
                        input_val, input_surface_val, target_val, target_surface_val = input_val.to(
                            device), input_surface_val.to(device), target_val.to(device), target_surface_val.to(device)

                        # Inference
                        output_val, output_surface_val = model(input_val, input_surface_val,
                                                               aux_constants['weather_statistics'],
                                                               aux_constants['constant_maps'], aux_constants['const_h'])
                        # Noralize the gt to make the loss compariable
                        target_val, target_surface_val = utils_data.normData(target_val, target_surface_val,
                                                                             aux_constants['weather_statistics_last'])

                        val_loss_surface = criterion(
                            output_surface_val, target_surface_val)
                        weighted_val_loss_surface = torch.mean(
                            val_loss_surface * surface_weights)

                        val_loss_upper = criterion(output_val, target_val)
                        weighted_val_loss_upper = torch.mean(
                            val_loss_upper * upper_weights)

                        loss = weighted_val_loss_upper + weighted_val_loss_surface * 0.25

                        val_loss += loss.item()

                    val_loss /= len(val_loader)
                    writer.add_scalars('Loss',
                                       {'train': epoch_loss,
                                        'val': val_loss},
                                       i)
                    logger.info(
                        "Validate at Epoch {} : {:.3f}".format(i, val_loss))

                    # Visualize the training process
                    if visualize:
                        png_path = os.path.join(res_path, "png_training")
                        utils.mkdirs(png_path)
                        
                        # Normalize the data back to the original space for visualization
                        output_val, output_surface_val = utils_data.normBackData(output_val, output_surface_val,
                                                                                aux_constants['weather_statistics_last'])
                        target_val, target_surface_val = utils_data.normBackData(target_val, target_surface_val,
                                                                                aux_constants['weather_statistics_last'])
                        
                        utils.visuailze(output_val.detach().cpu().squeeze(),
                                        target_val.detach().cpu().squeeze(),
                                        # input_val_raw.squeeze(),
                                        input_val.detach().cpu().squeeze(),
                                        var='u',
                                        z=12,
                                        step=i,
                                        path=png_path)
                        utils.visuailze_surface(output_surface_val.detach().cpu().squeeze(),
                                                target_surface_val.detach().cpu().squeeze(),
                                                # input_surface_val_raw.squeeze(),
                                                input_surface_val.detach().cpu().squeeze(),
                                                var='msl',
                                                step=i,
                                                path=png_path)
                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = copy.deepcopy(model)
                        # Save the best model
                        torch.save(best_model, os.path.join(
                            model_save_path, 'best_model.pth'))
                        logger.info(
                            f"current best model is saved at {i} epoch.")
                        epochs_since_last_improvement = 0
                    else:
                        epochs_since_last_improvement += 1
                        if epochs_since_last_improvement >= 5:  # TODO may move to config.py
                            logger.info(
                                f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training.")
                            break
                        
                    del input_val, input_surface_val, target_val, target_surface_val, output_val, output_surface_val
                    torch.cuda.empty_cache()

        if rank == 0:
            print("lr:", lr_scheduler.get_last_lr()[0])
    return best_model


def test(test_loader, model, device, res_path, visualize=False):
    # set up empty dics for rmses and anormaly correlation coefficients
    rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v = dict(
    ), dict(), dict(), dict(), dict()
    rmse_surface = dict()

    acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(
    ), dict(), dict(), dict(), dict()
    acc_surface = dict()

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device)
    
    # Loss function
    criterion = nn.L1Loss(reduction='none')
    upper_weights, surface_weights = aux_constants['variable_weights']
    test_loss = 0.0
    
    batch_id = 0
    # for id, data in enumerate(test_loader, 0):
    for data in tqdm(test_loader, desc='Testing'):
        # Store initial input for different models
        # print(f"predict on {id}")
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        input_test, input_surface_test, target_test, target_surface_test = \
            input_test.to(device), input_surface_test.to(
                device), target_test.to(device), target_surface_test.to(device)
        model.eval()

        # Inference
        output_test, output_surface_test = model(input_test, input_surface_test,
                                                 aux_constants['weather_statistics'],
                                                 aux_constants['constant_maps'], aux_constants['const_h'])
        
        # Noralize the gt to make the loss compariable
        target_test_normalized, target_surface_test_normalized = utils_data.normData(target_test, target_surface_test,
                                                            aux_constants['weather_statistics_last'])
        
        test_loss_surface = criterion(
            output_surface_test, target_surface_test_normalized)
        weighted_test_loss_surface = torch.mean(
            test_loss_surface * surface_weights)

        test_loss_upper = criterion(output_test, target_test_normalized)
        weighted_test_loss_upper = torch.mean(
            test_loss_upper * upper_weights)

        loss = weighted_test_loss_upper + weighted_test_loss_surface * 0.25

        test_loss += loss.item()
        
        # Transfer to the output to the original data range
        output_test, output_surface_test = utils_data.normBackData(output_test, output_surface_test,
                                                                   aux_constants['weather_statistics_last'])

        target_time = periods_test[1][batch_id]

        # Visualize
        if visualize:
            png_path = os.path.join(res_path, "png")
            utils.mkdirs(png_path)

            utils.visuailze(output_test.detach().cpu().squeeze(),
                            target_test.detach().cpu().squeeze(),
                            input_test.detach().cpu().squeeze(),
                            var='t',
                            z=2,
                            step=target_time,
                            path=png_path)
            # ['msl', 'u','v','t2m']
            utils.visuailze_surface(output_surface_test.detach().cpu().squeeze(),
                                    target_surface_test.detach().cpu().squeeze(),
                                    input_surface_test.detach().cpu().squeeze(),
                                    var='u10',
                                    step=target_time,
                                    path=png_path)
            utils.visuailze_surface(output_surface_test.detach().cpu().squeeze(),
                                    target_surface_test.detach().cpu().squeeze(),
                                    input_surface_test.detach().cpu().squeeze(),
                                    var='v10',
                                    step=target_time,
                                    path=png_path)

        # Compute test scores
        # rmse
        output_test = output_test.squeeze()
        target_test = target_test.squeeze()
        output_surface_test = output_surface_test.squeeze()
        target_surface_test = target_surface_test.squeeze()

        rmse_upper_z[target_time] = score.weighted_rmse_torch_channels(output_test[0],
                                                                       target_test[0]).detach().cpu().numpy()
        rmse_upper_q[target_time] = score.weighted_rmse_torch_channels(output_test[1],
                                                                       target_test[1]).detach().cpu().numpy()
        rmse_upper_t[target_time] = score.weighted_rmse_torch_channels(output_test[2],
                                                                       target_test[2]).detach().cpu().numpy()
        rmse_upper_u[target_time] = score.weighted_rmse_torch_channels(output_test[3],
                                                                       target_test[3]).detach().cpu().numpy()
        rmse_upper_v[target_time] = score.weighted_rmse_torch_channels(output_test[4],
                                                                       target_test[4]).detach().cpu().numpy()

        rmse_surface[target_time] = score.weighted_rmse_torch_channels(output_surface_test,
                                                                       target_surface_test).detach().cpu().numpy()

        # acc
        surface_mean, _, upper_mean, _ = aux_constants['weather_statistics_last']
        output_test_anomaly = output_test - upper_mean.squeeze(0)
        output_surface_test_anomaly = output_surface_test - \
            surface_mean.squeeze(0)
        target_test_anomaly = target_test - upper_mean.squeeze(0)
        target_surface_test_anomaly = target_surface_test - \
            surface_mean.squeeze(0)

        acc_upper_z[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[0],
                                                                     target_test_anomaly[0]).detach().cpu().numpy()
        acc_upper_q[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[1],
                                                                     target_test_anomaly[1]).detach().cpu().numpy()
        acc_upper_t[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[2],
                                                                     target_test_anomaly[2]).detach().cpu().numpy()
        acc_upper_u[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[3],
                                                                     target_test_anomaly[3]).detach().cpu().numpy()
        acc_upper_v[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[4],
                                                                     target_test_anomaly[4]).detach().cpu().numpy()

        acc_surface[target_time] = score.weighted_acc_torch_channels(output_surface_test_anomaly,
                                                                     target_surface_test_anomaly).detach().cpu().numpy()
        
    # Save rmses to csv
    csv_path = os.path.join(res_path, "csv")
    utils.mkdirs(csv_path)
    utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_surface,
                           "rmse")
    utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q,
                           acc_upper_t, acc_upper_u, acc_upper_v, acc_surface, "acc")
    
    test_loss /= len(test_loader)
    print('test_loss:', test_loss)


if __name__ == "__main__":
    pass
