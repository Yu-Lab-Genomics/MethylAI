import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import datetime
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.utils import check_output_folder

class MethylAITrainer:
    def __init__(self, model: nn.Module, optimizer: optim.AdamW, scheduler: SequentialLR,
                 train_dataloader: DataLoader, validation_dataloader: DataLoader,
                 max_step_per_epoch: int, pretrain_snapshot_path: str, snapshot_path: str,
                 is_reverse_complement_augmentation: bool,
                 is_load_output_block_pretrain_weight: bool,
                 is_run_validation_at_first: bool,
                 is_quiet: bool,
                 output_folder: str, output_result_file: str,
                 save_model_epoch_number=1, minimal_loss_weight_for_validation=1.0,
                 print_loss_step=500, print_model_output_step=5000):
        # GPU id & device & word size
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.device = torch.device(self.gpu_id)
        self.word_size = int(os.environ['WORLD_SIZE'])
        # 模型
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        # optimizer & scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        # train & validation dataloader & is_reverse_complement_augmentation
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation
        # 如果指定了snapshot_path，则加载该snapshot
        self.pretrain_snapshot_path = pretrain_snapshot_path
        self.snapshot_path = snapshot_path
        self.is_load_output_block_pretrain_weight = is_load_output_block_pretrain_weight
        self._load_snapshot()
        # 是否需要先运行模型验证
        self.is_run_validation_at_first = is_run_validation_at_first
        self.is_verbose = not is_quiet
        # loss function
        self.mse_loss = nn.MSELoss(reduction='none')
        self.huber_loss = nn.HuberLoss(reduction='none', delta=0.15)
        # loss value
        self.smooth_loss_item = 0.0
        self.raw_loss_item = 0.0
        self.window_loss_item = 0.0
        self.combined_loss_item = 0.0
        # epoch number
        self.running_epoch_number = 0
        self.max_step_per_epoch = max_step_per_epoch
        # 保存输出文件的文件夹 & 检查文件夹是否存在
        self.output_folder = output_folder
        self.snapshot_folder = f'{self.output_folder}/snapshot'
        if self.gpu_id == 0:
            check_output_folder(self.output_folder)
            check_output_folder(self.snapshot_folder)
        # 模型训练和验证结果文件
        self.output_result_file = f'{self.output_folder}/{output_result_file}'
        # 保存每个epoch的结果(epoch_number, train_loss, test_loss, PCC, SCC)，完成epoch时输出
        self.epoch_output_list = []
        # 用于validation最低的loss_weight
        self.minimal_loss_weight_for_validation = minimal_loss_weight_for_validation
        # 多少epoch保存一次模型
        self.save_model_epoch_number = save_model_epoch_number
        # print相关设置
        self.print_loss_step = print_loss_step
        self.print_model_output_step = print_model_output_step

    def _load_snapshot(self):
        if self.snapshot_path:
            print('load snapshot: ', self.snapshot_path)
            all_state = torch.load(self.snapshot_path, map_location=self.device, weights_only=False)
            # 每次训练完再保存，因此加载的时候running_epoch_number + 1
            self.running_epoch_number = all_state['self.running_epoch_number'] + 1
            self.model.module.load_state_dict(all_state['self.model'])
            self.optimizer.load_state_dict(all_state['self.optimizer'])
            self.scheduler.load_state_dict(all_state['self.scheduler'])
        elif self.pretrain_snapshot_path:
            print('load pretrain snapshot: ', self.pretrain_snapshot_path)
            pretrain_all_state = torch.load(self.pretrain_snapshot_path, map_location=self.device, weights_only=False)
            pretrain_model_state = pretrain_all_state['self.module']
            model_state_dict = self.model.module.state_dict()
            for key in model_state_dict.keys():
                if key.startswith('input_block'):
                    if 'conv1_' in key:
                        model_state_dict[key] = pretrain_model_state[key]
                        if self.gpu_id == 0:
                            print(key)
                    elif 'bn1' in key:
                        pretrain_key = key.replace('bn1', 'bn_dict.human')
                        model_state_dict[key] = pretrain_model_state[pretrain_key]
                        if self.gpu_id == 0:
                            print(f'key: {key}; pretrain_key: {pretrain_key}')
                    elif 'conv_block' in key:
                        pretrain_key = key.replace('conv_block', 'conv_block_dict.human')
                        model_state_dict[key] = pretrain_model_state[pretrain_key]
                        if self.gpu_id == 0:
                            print(f'key: {key}; pretrain_key: {pretrain_key}')
                if key.startswith('body'):
                    if 'basic_block' in key:
                        pretrain_key = key.replace('basic_block', 'basic_block_dict.human')
                        model_state_dict[key] = pretrain_model_state[pretrain_key]
                        if self.gpu_id == 0:
                            print(f'key: {key}; pretrain_key: {pretrain_key}')
                    elif 'shortcut_conv' in key:
                        model_state_dict[key] = pretrain_model_state[key]
                        if self.gpu_id == 0:
                            print(key)
                # 如果预训练与微调使用相同的数据集，则加载output_block的权重
                if key.startswith('output_block') and self.is_load_output_block_pretrain_weight:
                    pretrain_key = key.replace('.output_block.', '.output_block_dict.human.')
                    model_state_dict[key] = pretrain_model_state[pretrain_key]
                    if self.gpu_id == 0:
                        print(f'key: {key}; pretrain_key: {pretrain_key}')
            self.model.module.load_state_dict(model_state_dict)
        else:
            print('Do NOT load snapshot!')

    def _save_snapshot(self, epoch_number):
        output_file_name = f'{self.snapshot_folder}/snapshot_epoch_{epoch_number}.pth'
        torch.save({
            'self.running_epoch_number': epoch_number,
            'self.model' : self.model.module.state_dict(),
            'self.optimizer' : self.optimizer.state_dict(),
            'self.scheduler' : self.scheduler.state_dict()
        }, output_file_name)
        print('Saved snapshot to', output_file_name)

    def _loss_function(self, prediction: torch.tensor, target: torch.tensor, weight: torch.tensor):
        # 计算raw、window输出值的下标
        output_dim = prediction.shape[1]
        raw_loss_start = output_dim // 5
        window_1000_loss_start = raw_loss_start * 2
        window_500_loss_start = raw_loss_start * 3
        window_200_loss_start = raw_loss_start * 4
        # smooth loss
        smooth_loss = (self.mse_loss(
            prediction[:, :raw_loss_start],
            target[:, :raw_loss_start]
        ) * weight[:, :raw_loss_start]).mean()
        # raw loss
        raw_loss = (self.huber_loss(
            prediction[:, raw_loss_start: window_1000_loss_start],
            target[:, raw_loss_start: window_1000_loss_start]
        ) * weight[:, raw_loss_start: window_1000_loss_start]).mean()
        # window loss
        window_1000_loss = (self.mse_loss(
            prediction[:, window_1000_loss_start: window_500_loss_start],
            target[:, window_1000_loss_start: window_500_loss_start]
        ) * weight[:, window_1000_loss_start: window_500_loss_start]).mean()
        window_500_loss = (self.mse_loss(
            prediction[:, window_500_loss_start: window_200_loss_start],
            target[:, window_500_loss_start: window_200_loss_start]
        ) * weight[:, window_500_loss_start: window_200_loss_start]).mean()
        window_200_loss = (self.mse_loss(
            prediction[:, window_200_loss_start:],
            target[:, window_200_loss_start:]
        ) * weight[:, window_200_loss_start:]).mean()
        window_loss = window_1000_loss + window_500_loss + window_200_loss
        combined_loss = smooth_loss + raw_loss + window_loss
        return combined_loss, smooth_loss.item(), raw_loss.item(), window_loss.item()

    def train_model(self, total_epoch_number):
        # 先运行验证
        if self.is_run_validation_at_first:
            self._validation_epoch(self.running_epoch_number - 1)
        # 进行训练
        for epoch_number in range(self.running_epoch_number, total_epoch_number):
            # train
            self._train_epoch(epoch_number)
            # save snapshot
            if self.gpu_id == 0 and (epoch_number % self.save_model_epoch_number == 0):
                self._save_snapshot(epoch_number)
            # validation
            self._validation_epoch(epoch_number)

    def _train_epoch(self, epoch_number):
        # batch_size & total step & total size
        batch_size = len(next(iter(self.train_dataloader))[0])
        dataloader_len = len(self.train_dataloader)
        total_step = min(self.max_step_per_epoch, dataloader_len)
        # 把loss值重置为0
        self.smooth_loss_item = 0.0
        self.raw_loss_item = 0.0
        self.window_loss_item = 0.0
        self.combined_loss_item = 0.0
        # 设置为训练模式
        self.model.train()
        # 打印训练信息
        print(f'Train: [GPU{self.gpu_id}] Epoch: {epoch_number} | Batchsize: {batch_size} | Steps: {total_step} | dataloader_len: {dataloader_len}')
        # 给train_dataloader设置epoch_number
        self.train_dataloader.sampler.set_epoch(epoch_number)
        # 记录开始时间
        train_start_time = datetime.datetime.now()
        # 开始训练
        for batch_index, (dna_one_hot, methylation_level, loss_weight) in enumerate(self.train_dataloader):
            combined_loss_item, smooth_loss_item, raw_loss_item, window_loss_item =\
                self._train_batch(dna_one_hot, methylation_level, loss_weight, batch_index)
            # 记录各个loss
            self.smooth_loss_item = self.smooth_loss_item + smooth_loss_item
            self.raw_loss_item = self.raw_loss_item + raw_loss_item
            self.window_loss_item = self.window_loss_item + window_loss_item
            self.combined_loss_item = self.combined_loss_item + combined_loss_item
            # 每N batch_index存打印一次训练过程
            if self.gpu_id == 0 and batch_index % self.print_loss_step == 0 and self.is_verbose:
                # 获取学习率
                learning_rate = self.scheduler.get_last_lr()
                # 计算当前所用的时间
                using_time = datetime.datetime.now() - train_start_time
                # print上述信息
                print(f'combined_loss: {combined_loss_item:.4f}, smooth_loss: {smooth_loss_item:.4f}, raw_loss: {raw_loss_item:.4f}, '
                      f'window_loss: {window_loss_item:.4f} | step: [{batch_index:6d}|{total_step:6d}]')
                print(f'learning rate: {learning_rate}')
                print(f'train using time is: {using_time}\n')
            # 超过训练步数则结束该epoch
            if batch_index > self.max_step_per_epoch:
                break
        # 训练结束
        # 计算平均loss
        self.smooth_loss_item = self.smooth_loss_item / total_step
        self.raw_loss_item = self.raw_loss_item / total_step
        self.window_loss_item = self.window_loss_item / total_step
        self.combined_loss_item = self.combined_loss_item / total_step
        # dist.all_gather收集所有train_loss
        all_gather_smooth_loss = self._all_gather_loss(self.smooth_loss_item)
        all_gather_raw_loss = self._all_gather_loss(self.raw_loss_item)
        all_gather_window_loss = self._all_gather_loss(self.window_loss_item)
        all_gather_combined_loss = self._all_gather_loss(self.combined_loss_item)
        # 由0号进程保存 epoch_number, train loss 到 self.epoch_output_list
        if self.gpu_id == 0:
            print(f'train: smooth_loss: {all_gather_smooth_loss:4f}, raw_loss: {all_gather_raw_loss:4f}, '
                  f'window_loss: {all_gather_window_loss:4f}, combined_loss: {all_gather_combined_loss:4f}')
            self.epoch_output_list.extend(['epoch number:', epoch_number])
            self.epoch_output_list.extend(
                ['train', 'smooth_loss', all_gather_smooth_loss, 'raw_loss', all_gather_raw_loss,
                 'window_loss', all_gather_window_loss, 'combined_loss', all_gather_combined_loss]
            )
            self._write_epoch_output(epoch_is_finish=False)
            # 计算并打印训练时间
            train_time = datetime.datetime.now() - train_start_time
            print(f'train epoch time is: {train_time}')

    def _train_batch(self, dna_one_hot, methylation_level, loss_weight, batch_index):
        # move to self.device
        dna_one_hot_tensor = dna_one_hot.to(self.device)
        methylation_level_tensor = methylation_level.to(self.device)
        loss_weight_tensor = loss_weight.to(self.device)
        # Compute prediction error
        model_output = self.model(dna_one_hot_tensor)
        combined_loss, smooth_loss_item, raw_loss_item, window_loss_item =\
            self._loss_function(model_output, methylation_level_tensor, loss_weight_tensor)
        # Backpropagation & optimize
        self.optimizer.zero_grad()
        combined_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # 打印具体信息
        if self.gpu_id == 0 and batch_index % self.print_model_output_step == 0  and self.is_verbose:
            torch.set_printoptions(precision=3, threshold=160, edgeitems=4, sci_mode=False)
            print(f'dna_one_hot: {dna_one_hot.shape}')
            print(dna_one_hot.detach())
            print(f'model_output: {model_output.shape}')
            print(model_output.detach())
            print(f'methylation_level: {methylation_level_tensor.shape}')
            print(methylation_level_tensor.detach())
            print(f'loss_weight: {loss_weight_tensor.shape}')
            print(loss_weight_tensor.detach())
        return combined_loss.item(), smooth_loss_item, raw_loss_item, window_loss_item

    def _validation_epoch(self, epoch_number):
        # total step & total size
        batch_size = len(next(iter(self.train_dataloader))[0])
        total_step = len(self.validation_dataloader)
        total_size = len(self.validation_dataloader.sampler)
        # 把loss值重置为0
        self.smooth_loss_item = 0.0
        self.raw_loss_item = 0.0
        self.window_loss_item = 0.0
        self.combined_loss_item = 0.0
        # 以下变量分别用于保存模型输出值、真实值、loss权重
        model_output_list = []
        true_list = []
        loss_weight_list = []
        # 设置为验证模式
        self.model.eval()
        # 打印训练信息
        print(f'Validation: [GPU{self.gpu_id}] Epoch: {epoch_number} | Batchsize: {batch_size} | Steps: {total_step} | Sizes: {total_size}')
        # 记录开始时间
        validation_start_time = datetime.datetime.now()
        # 开始运行
        # 即self.is_reverse_complement_augmentation == True
        if self.is_reverse_complement_augmentation:
            with torch.no_grad():
                for batch_index, (forward_dna_one_hot, reverse_dna_one_hot, methylation_level, loss_weight
                                  ) in enumerate(self.validation_dataloader):
                    forward_dna_sequence_one_hot_tensor = forward_dna_one_hot.to(self.gpu_id)
                    reverse_dna_sequence_one_hot_tensor = reverse_dna_one_hot.to(self.gpu_id)
                    methylation_level_tensor = methylation_level.to(self.gpu_id)
                    loss_weight_tensor = loss_weight.to(self.gpu_id)
                    # 计算output
                    forward_output_tensor = self.model(forward_dna_sequence_one_hot_tensor)
                    reverse_output_tensor = self.model(reverse_dna_sequence_one_hot_tensor)
                    average_output_tensor = (forward_output_tensor.detach() + reverse_output_tensor.detach()) / 2
                    # 记录各个loss
                    combined_loss, smooth_loss_item, raw_loss_item, window_loss_item =\
                        self._loss_function(average_output_tensor, methylation_level_tensor, loss_weight_tensor)
                    self.smooth_loss_item = self.smooth_loss_item + smooth_loss_item
                    self.raw_loss_item = self.raw_loss_item + raw_loss_item
                    self.window_loss_item = self.window_loss_item + window_loss_item
                    self.combined_loss_item = self.combined_loss_item + combined_loss.item()
                    # dist.all_gather所有进程的average_output_tensor、methylation_level_tensor、loss_weight_tensor
                    average_output_tensor_list = self._all_gather_tensor(average_output_tensor)
                    methylation_level_tensor_list = self._all_gather_tensor(methylation_level_tensor)
                    loss_weight_tensor_list = self._all_gather_tensor(loss_weight_tensor)
                    # 上述tensor由0号进程保存
                    if self.gpu_id == 0:
                        all_gather_average_output_tensor = torch.concatenate(average_output_tensor_list, dim=0)
                        all_gather_methylation_level_tensor = torch.concatenate(methylation_level_tensor_list, dim=0)
                        all_gather_loss_weight_tensor = torch.concatenate(loss_weight_tensor_list, dim=0)
                        # 把tensor转换为numpy，并保存到对应的list中
                        model_output_list.append(all_gather_average_output_tensor.cpu().numpy())
                        true_list.append(all_gather_methylation_level_tensor.cpu().numpy())
                        loss_weight_list.append(all_gather_loss_weight_tensor.cpu().numpy())
                    if self.gpu_id == 0 and batch_index % self.print_loss_step == 0 and self.is_verbose:
                        # 打印用时信息
                        print(f'validation batch index: {batch_index:5d}|{total_step:5d} (reverse complement mode)')
                        using_time = datetime.datetime.now() - validation_start_time
                        print(f'validation using time is: {using_time}\n')
                    if self.gpu_id == 0 and batch_index % self.print_model_output_step == 0 and self.is_verbose:
                        print(f'all_gather_output_tensor:\n{all_gather_average_output_tensor}')
                        print(f'all_gather_methylation_level_tensor:\n{all_gather_methylation_level_tensor}')
                        print(f'all_gather_loss_weight_tensor:\n{all_gather_loss_weight_tensor}')
        # 即self.is_reverse_complement_augmentation == False
        else:
            with torch.no_grad():
                for batch_index, (dna_sequence_one_hot_encoding, methylation_level, loss_weight
                                  ) in enumerate(self.validation_dataloader):
                    dna_sequence_one_hot_encoding_tensor = dna_sequence_one_hot_encoding.to(self.device)
                    methylation_level_tensor = methylation_level.to(self.device)
                    loss_weight_tensor = loss_weight.to(self.device)
                    output_tensor = self.model(dna_sequence_one_hot_encoding_tensor).detach()
                    # 记录各个loss
                    combined_loss, smooth_loss_item, raw_loss_item, window_loss_item = \
                        self._loss_function(output_tensor, methylation_level_tensor, loss_weight_tensor)
                    self.smooth_loss_item = self.smooth_loss_item + smooth_loss_item
                    self.raw_loss_item = self.raw_loss_item + raw_loss_item
                    self.window_loss_item = self.window_loss_item + window_loss_item
                    self.combined_loss_item = self.combined_loss_item + combined_loss.item()
                    # dist.all_gather所有进程的average_output_tensor、methylation_level_tensor、loss_weight_tensor
                    output_tensor_list = self._all_gather_tensor(output_tensor)
                    methylation_level_tensor_list = self._all_gather_tensor(methylation_level_tensor)
                    loss_weight_tensor_list = self._all_gather_tensor(loss_weight_tensor)
                    # 上述tensor由0号进程保存
                    if self.gpu_id == 0:
                        all_gather_output_tensor = torch.concatenate(output_tensor_list, dim=0)
                        all_gather_methylation_level_tensor = torch.concatenate(methylation_level_tensor_list, dim=0)
                        all_gather_loss_weight_tensor = torch.concatenate(loss_weight_tensor_list, dim=0)
                        # 把tensor转换为numpy，并保存到对应的list中
                        model_output_list.append(all_gather_output_tensor.cpu().numpy())
                        true_list.append(all_gather_methylation_level_tensor.cpu().numpy())
                        loss_weight_list.append(all_gather_loss_weight_tensor.cpu().numpy())
                    if self.gpu_id == 0 and batch_index % self.print_loss_step == 0 and self.is_verbose:
                        # 打印用时信息
                        print(f'validation batch index: {batch_index:5d}|{total_step:5d}')
                        using_time = datetime.datetime.now() - validation_start_time
                        print(f'validation using time is: {using_time}\n')
                    if self.gpu_id == 0 and batch_index % self.print_model_output_step == 0 and self.is_verbose:
                        print(f'all_gather_output_tensor:\n{all_gather_output_tensor}')
                        print(f'all_gather_methylation_level_tensor:\n{all_gather_methylation_level_tensor}')
                        print(f'all_gather_loss_weight_tensor:\n{all_gather_loss_weight_tensor}')
        # 计算平均loss
        self.smooth_loss_item = self.smooth_loss_item / total_step
        self.raw_loss_item = self.raw_loss_item / total_step
        self.window_loss_item = self.window_loss_item / total_step
        self.combined_loss_item = self.combined_loss_item / total_step
        # dist.all_gather收集所有validation loss
        all_gather_smooth_loss = self._all_gather_loss(self.smooth_loss_item)
        all_gather_raw_loss = self._all_gather_loss(self.raw_loss_item)
        all_gather_window_loss = self._all_gather_loss(self.window_loss_item)
        all_gather_combined_loss = self._all_gather_loss(self.combined_loss_item)
        # 由0号进程保存完成PCC、SCC的计算和输出
        if self.gpu_id == 0:
            # 打印并保存all_gather_validation_loss
            print(f'validation smooth_loss: {all_gather_smooth_loss:4f}, raw_loss: {all_gather_raw_loss:4f}, '
                  f'window_loss: {all_gather_window_loss:4f}, combined_loss: {all_gather_combined_loss:4f}')
            self.epoch_output_list.extend(
                ['validation', 'smooth_loss', all_gather_smooth_loss, 'raw_loss', all_gather_raw_loss,
                 'window_loss', all_gather_window_loss, 'combined_loss', all_gather_combined_loss]
            )
            # 以下numpy用于计算每套数据的PCC、SCC
            prediction_numpy = np.concatenate(model_output_list, axis=0).T
            true_numpy = np.concatenate(true_list, axis=0).T
            loss_weight_numpy = np.concatenate(loss_weight_list, axis=0).T
            # 计算PCC、SCC并输出
            self._calculate_pcc_scc(prediction_numpy, true_numpy, loss_weight_numpy)
            self._write_epoch_output(epoch_is_finish=True)
            # 打印用时信息
            validation_time = datetime.datetime.now() - validation_start_time
            print(f'validation time:{validation_time}\n')
        # 等待所有进程完成
        dist.barrier()

    def _all_gather_tensor(self, tensor: torch.tensor):
        tensor_list = [torch.zeros(tensor.shape, device=self.device, dtype=tensor.dtype) for _ in range(self.word_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list

    def _all_gather_loss(self, loss_value):
        loss_tensor = torch.tensor([loss_value], device=self.device)
        loss_tensor_list = self._all_gather_tensor(loss_tensor)
        all_gather_loss = torch.mean(torch.concatenate(loss_tensor_list)).item()
        return all_gather_loss

    def _calculate_pcc_scc(self, prediction_numpy: np.ndarray, true_numpy: np.ndarray, loss_weight_numpy: np.ndarray):
        number_dataset = true_numpy.shape[0]
        pcc_list = ['PCC']
        scc_list = ['SCC']
        for i in range(number_dataset):
            predict_value = prediction_numpy[i]
            true_value = true_numpy[i]
            loss_weight_value = loss_weight_numpy[i]
            # 根据loss_weight筛选用于计算PCC和SCC的数据
            true_value_index = (loss_weight_value >= self.minimal_loss_weight_for_validation)
            predict_value = predict_value[true_value_index]
            true_value = true_value[true_value_index]
            pearson_corr, _ = pearsonr(predict_value, true_value)
            spearman_corr, _ = spearmanr(predict_value, true_value)
            pcc_list.append(pearson_corr)
            scc_list.append(spearman_corr)
        self.epoch_output_list.extend(pcc_list)
        self.epoch_output_list.extend(scc_list)

    def _write_epoch_output(self, epoch_is_finish: bool):
        # 把self.epoch_output_list中所有float、int转换为str格式，float保留小数点后4位，
        epoch_output_line = '\t'.join(
            [str(x) if isinstance(x, (int, str)) else "{:.4f}".format(x) for x in self.epoch_output_list])
        if epoch_is_finish:
            epoch_output_line = epoch_output_line + '\n'
        else:
            epoch_output_line = epoch_output_line + '\t'
        # 然后把self.epoch_output_list清空
        self.epoch_output_list = []
        with open(self.output_result_file, 'a') as file:
            file.write(epoch_output_line)
