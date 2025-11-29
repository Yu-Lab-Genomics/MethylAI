data_folder = ''
checkpoint_folder = ''
output_folder = ''

methylai_parameter_dict = {
    # trainer
    'total_epoch_number': 3,
    'max_step_per_epoch': 10_000_000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0006,
    'weight_decay': 0.01,
    'batch_size': 150,
    'output_folder': output_folder,
    'pretrain_snapshot_path': f'{checkpoint_folder}/1_train/snapshot/snapshot_epoch_1.pth',
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
    'output_block_dims': (96 * 80, 96 * 20, 96 * 5),
    # dataset & dataloader设置
    'input_dna_length': 9 * 2**11,
    'train_num_workers': 2,
    'validation_num_workers': 4,
    'loss_weight_factor': 5,
    'max_loss_weight_factor': 1.0,
    'minimal_coverage': 5,
    'keep_raw_methylation': True,
    'reverse_complement_augmentation': True,
    # learning rate scheduler
    'warmup_lr_epoch_number': 1,
    'constant_lr_epoch_number': 1,
    # train/validation set
    'train_set_file': '/home/chenfaming/pool2/project/231009_DNA_methylation_data/241211_human_data/17_human_ENCODE_dataset/2_dataset_file/'
                      'human_encode_smooth_raw_window3_train_chromosome.pkl',
    'bed_dataset_to_repetition_dict': {
        '/home/chenfaming/pool2/project/231009_DNA_methylation_data/241211_human_data/17_human_ENCODE_dataset/2_dataset_file/'
        'human_encode_smooth_raw_window3_train_chromosome_coordinate.inter_CGI_2.bed': 2,
    },
    'validation_set_file': '/home/chenfaming/pool2/project/231009_DNA_methylation_data/241211_human_data/17_human_ENCODE_dataset/2_dataset_file/'
                           'human_encode_smooth_raw_window3_validation_chromosome.pkl',
    'genome_fasta_file': '/home/chenfaming/genome/ucsc_hg38/hg38.fa',
}

