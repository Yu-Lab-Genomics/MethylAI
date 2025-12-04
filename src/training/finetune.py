import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
from datetime import timedelta
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.dataset.training_dataset import MethylAITrainDataset
from MethylAI.src.dataset.validation_dataset import MethylAIValidationDataset
from MethylAI.src.model.methylai import MethylAI
from MethylAI.src.trainer.trainer import MethylAITrainer
from MethylAI.src.utils.utils import load_config

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend='nccl', timeout=timedelta(minutes=120))

def initialize_train_validation_dataset_dataloader(dataset_parameter_dict: dict):
    train_dataset = MethylAITrainDataset(
        dataset_file=dataset_parameter_dict['train_set_file'],
        bed_dataset_to_repetition_dict=dataset_parameter_dict['cpg_index_to_repetition_dict'],
        genome_fasta_file=dataset_parameter_dict['genome_fasta_file'],
        model_input_dna_length=dataset_parameter_dict['input_dna_length'],
        loss_weight_factor=dataset_parameter_dict['loss_weight_factor'],
        max_loss_weight_factor=dataset_parameter_dict['max_loss_weight_factor'],
        minimal_coverage=dataset_parameter_dict['minimal_coverage'],
        is_reverse_complement_augmentation=dataset_parameter_dict['is_reverse_complement_augmentation'],
        is_keep_raw_methylation=dataset_parameter_dict['is_keep_raw_methylation'],
    )
    validation_dataset = MethylAIValidationDataset(
        dataset_file=dataset_parameter_dict['validation_set_file'],
        genome_fasta_file=dataset_parameter_dict['genome_fasta_file'],
        model_input_dna_length=dataset_parameter_dict['input_dna_length'],
        loss_weight_factor=dataset_parameter_dict['loss_weight_factor'],
        max_loss_weight_factor=dataset_parameter_dict['max_loss_weight_factor'],
        minimal_coverage=dataset_parameter_dict['minimal_coverage'],
        is_reverse_complement_augmentation=dataset_parameter_dict['is_reverse_complement_augmentation'],
        is_keep_raw_methylation=dataset_parameter_dict['is_keep_raw_methylation'],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=dataset_parameter_dict['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=dataset_parameter_dict['train_num_workers'],
        sampler=DistributedSampler(train_dataset),
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=dataset_parameter_dict['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=dataset_parameter_dict['validation_num_workers'],
        sampler=DistributedSampler(
            dataset=validation_dataset,
            shuffle=False
        ),
    )
    print(f'training set size: {len(train_dataloader.dataset)}')
    print(f'validation set size: {len(validation_dataloader.dataset)}')
    return train_dataloader, validation_dataloader

def initialize_optimizer_scheduler(model: nn.Module, model_parameter_dict: dict, step_per_epoch: int):
    # from: https://pytorch.org/docs/stable/optim.html
    cnn_bias_parameter_group = [
        p for name, p in model.named_parameters() if ('output_block' not in name) and ('bias' in name)
    ]
    cnn_weight_parameter_group = [
        p for name, p in model.named_parameters() if ('output_block' not in name) and ('bias' not in name)
    ]
    output_bias_parameter_group = [
        p for name, p in model.named_parameters() if ('output_block' in name) and ('bias' in name)
    ]
    output_weight_parameter_group = [
        p for name, p in model.named_parameters() if ('output_block' in name) and ('bias' not in name)
    ]
    assert len(list(model.parameters())) == (len(cnn_bias_parameter_group) + (len(cnn_weight_parameter_group)) +
                                             len(output_bias_parameter_group) + len(output_weight_parameter_group))
    output_block_learning_rate = model_parameter_dict['output_block_learning_rate']
    parameter_group = [
        {'params': cnn_bias_parameter_group, 'weight_decay': 0.0},
        {'params': cnn_weight_parameter_group},
        {'params': output_bias_parameter_group, 'lr': output_block_learning_rate, 'weight_decay': 0.0},
        {'params': output_weight_parameter_group, 'lr': output_block_learning_rate}
    ]
    # optimizer
    learning_rate = model_parameter_dict['learning_rate']
    weight_decay = model_parameter_dict['weight_decay']
    optimizer = optim.AdamW(
        parameter_group, lr=learning_rate, weight_decay=weight_decay
    )
    # scheduler
    warmup_lr_epoch_number = model_parameter_dict['warmup_lr_epoch_number']
    constant_lr_epoch_number = model_parameter_dict['constant_lr_epoch_number']
    warmup_step = int(step_per_epoch * warmup_lr_epoch_number)
    constant_step = int(step_per_epoch * constant_lr_epoch_number)
    # warmup_scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=(1.0 / warmup_step), total_iters=(warmup_step - 1)
    )
    # constant_scheduler
    constant_scheduler = ConstantLR(
        optimizer, factor=1, total_iters=constant_step
    )
    # CosineAnnealingWarmRestarts
    cosine_annealing_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=step_per_epoch, T_mult=2, eta_min=learning_rate * 0.001
    )
    # 把上述scheduler汇总
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler, cosine_annealing_scheduler],
        milestones=[warmup_step, warmup_step + constant_step]
    )
    return optimizer, scheduler

def main_train_model_argparse():
    # argparse
    parser = argparse.ArgumentParser()
    # required parameter
    parser.add_argument('--config_file', required=True,
                        help='Python configuration file defining model architecture and hyperparameters.')
    parser.add_argument('--config_dict_name', required=True,
                        help='Name of the Python dictionary variable containing configuration parameters.')
    # optional parameter
    parser.add_argument('--quite', action='store_true', help='')
    parser.add_argument('--print_loss_step', type=int, default=500, help='')
    parser.add_argument('--print_model_output_step', type=int, default=5000, help='')
    args = parser.parse_args()
    ddp_setup()
    methylai_parameter_dict = load_config(args.config_file, args.config_dict_name)
    methylai_model = MethylAI(methylai_parameter_dict)
    methylai_model = nn.SyncBatchNorm.convert_sync_batchnorm(methylai_model)
    train_dataloader, validation_dataloader = initialize_train_validation_dataset_dataloader(methylai_parameter_dict)
    step_per_epoch = min(methylai_parameter_dict['max_step_per_epoch'], len(train_dataloader))
    optimizer, scheduler = initialize_optimizer_scheduler(methylai_model, methylai_parameter_dict, step_per_epoch=step_per_epoch)
    model_trainer = MethylAITrainer(
        model=methylai_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        max_step_per_epoch=methylai_parameter_dict['max_step_per_epoch'],
        pretrain_snapshot_path=methylai_parameter_dict['pretrain_snapshot_path'],
        snapshot_path=methylai_parameter_dict['snapshot_path'],
        is_reverse_complement_augmentation=methylai_parameter_dict['is_reverse_complement_augmentation'],
        is_load_output_block_pretrain_weight=methylai_parameter_dict['is_load_output_block_pretrain_weight'],
        is_run_validation_at_first=methylai_parameter_dict['is_run_validation_at_first'],
        is_quiet=args.quite,
        save_model_epoch_number=1,
        minimal_loss_weight_for_validation=1.0,
        print_loss_step=args.print_loss_step,
        print_model_output_step=args.print_model_output_step,
        output_folder=methylai_parameter_dict['output_folder'],
        output_result_file=methylai_parameter_dict['output_result_file']
    )
    model_trainer.train_model(total_epoch_number=methylai_parameter_dict['total_epoch_number'])
    destroy_process_group()


if __name__ == "__main__":
    main_train_model_argparse()