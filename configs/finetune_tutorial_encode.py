methylai_config_dict = {
    # trainer
    'total_epoch_number': 3,
    'max_step_per_epoch': 10_000_000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0006,
    'weight_decay': 0.01,
    'batch_size': 50,
    'output_folder': '',
    'output_result_file': '',
    'pretrain_snapshot_path': '',
    'is_load_output_block_pretrain_weight': False,
    'snapshot_path': None,
    'is_run_validation_at_first': False,
    # model
    'input_block_channel': [20, 190, 30],
    'input_block_kernel_size': [3, 9, 21],
    'input_block_additional_conv_layer': (9, 9),
    'input_block_exponential_activation': True,
    'width': [300, 360, 420, 480, 540, 600],
    'depth': [2, 2, 2, 2, 2, 2],
    'kernel_size': [9, 9, 9, 9, 9, 9],
    'stride': [4, 4, 4, 4, 4, 2],
    'output_block_dims': (4 * 80, 4 * 5),
    # dataset & dataloader设置
    'input_dna_length': 9 * 2**11,
    'train_num_workers': 2,
    'validation_num_workers': 4,
    'loss_weight_factor': 5,
    'max_loss_weight_factor': 1.0,
    'minimal_coverage': 5,
    'is_keep_raw_methylation': True,
    'is_reverse_complement_augmentation': True,
    # learning rate scheduler
    'warmup_lr_epoch_number': 1,
    'constant_lr_epoch_number': 1,
    # train/validation set
    'train_set_file': '',
    'cpg_index_to_repetition_dict': {},
    'validation_set_file': '',
    'genome_fasta_file': '',
}

