

methylai_config_dict = {
    # trainer设置
    'total_epoch_number': 3,
    'max_step_per_epoch': 100_0000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0005,
    'weight_decay': 0.01,
    'batch_size': 50,
    'pretrain_snapshot_path': None,
    'is_load_output_block_pretrain_weight': True,
    'snapshot_path': None,
    'is_run_validation_at_first': False,
    'output_folder': '4_finetune_human1681',
    # model hyper parameter
    'input_block_channel': [20, 190, 30],
    'input_block_kernel_size': [3, 9, 21],
    'input_block_additional_conv_layer': (9, 9),
    'input_block_exponential_activation': True,
    'width': [300, 360, 420, 480, 540, 600],
    'depth': [2, 2, 2, 2, 2, 2],
    'kernel_size': [9, 9, 9, 9, 9, 9],
    'stride': [4, 4, 4, 4, 4, 2],
    'output_block_dims': (1574 * 5, 1574 * 5),
    # dataset & dataloader
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
    # dataset file path
    'train_set_file': '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
                      'human1681_smooth_raw_window3_train_chromosome',
    'bed_dataset_to_repetition_dict': {
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
        'human1681_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend500.uniq.bed': 1,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
        'human1681_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend250.uniq.bed': 2,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
        'human1681_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend100.uniq.bed': 2,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
        'human1681_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme.uniq.bed': 2,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
        'human1681_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_train_repetition.bed': 2,
    },
    'validation_set_file': '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/'
                           'human1681_smooth_raw_window3_validation_chromosome.pkl',
    'genome_fasta_file': '/home/chenfaming/genome/ucsc_hg38/hg38.fa',
}
