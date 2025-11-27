genome_fa_dir = '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241125_genome_fa'
species_dataset_dir = '/home/chenfaming/project/231009_DNA_methylation_data/241008_methbank_multi_species/9_smooth_raw_window3_dataset'
species_dict = {
    'human': {'genome_fa': f'{genome_fa_dir}/hg38.fa',
              'dataset': f'/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/human1681_smooth_raw_window3_train_chromosome',
              'output_block_dims': (5*1574, 5*1574)},
    'mouse': {'genome_fa': f'{genome_fa_dir}/mm10.fa',
              'dataset': '/home/chenfaming/project/231009_DNA_methylation_data/240924_ENCODE_mouse/4_smooth_raw_window3_dataset/mouse_encode_37k_smooth_raw_window3_train_chromosome',
              'output_block_dims': (10*72, 5*72)},
    '1_ail': {'genome_fa': f'{genome_fa_dir}/Ailuropoda_melanoleuca.ASM200744v2.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/1_Ailuropoda_melanoleuca_panda/ail_train_chromosome',
              'output_block_dims': (10*3, 5*3)},
    '2_bos': {'genome_fa': f'{genome_fa_dir}/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/2_Bos_taurus_cow/bos_train_chromosome',
              'output_block_dims': (10*129, 5*129)},
    '3_can': {'genome_fa': f'{genome_fa_dir}/Canis_lupus_familiaris.CanFam3.1.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/3_Canis_lupus_familiaris_dog/can_train_chromosome',
              'output_block_dims': (10*10, 5*10)},
    '4_gor': {'genome_fa': f'{genome_fa_dir}/Gorilla_gorilla.gorGor4.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/4_Gorilla_gorilla_gorilla/gor_train_chromosome',
              'output_block_dims': (10*4, 5*4)},
    '5_fas': {'genome_fa': f'{genome_fa_dir}/Macaca_fascicularis.Macaca_fascicularis_5.0.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/5_Macaca_fascicularis_macaque/fas_train_chromosome',
              'output_block_dims': (10*3, 5*3)},
    '6_mul': {'genome_fa': f'{genome_fa_dir}/Macaca_mulatta.Mmul_10.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/6_Macaca_mulatta_macaque/mul_train_chromosome',
              'output_block_dims': (10*7, 5*7)},
    '7_ovi': {'genome_fa': f'{genome_fa_dir}/Ovis_aries.Oar_v3.1.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/7_Ovis_aries_sheep/ovi_train_chromosome',
              'output_block_dims': (10*11, 5*11)},
    '8_pan': {'genome_fa': f'{genome_fa_dir}/Pan_troglodytes.Pan_tro_3.0.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/8_Pan_troglodytes_chimpanzee/pan_train_chromosome',
              'output_block_dims': (10*4, 5*4)},
    '9_rat': {'genome_fa': f'{genome_fa_dir}/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa',
              'dataset': f'{species_dataset_dir}/9_Rattus_norvegicus_rat/rat_train_chromosome',
              'output_block_dims': (10*3, 5*3)},
    '10_sus': {'genome_fa': f'{genome_fa_dir}/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa',
               'dataset': f'{species_dataset_dir}/10_Sus_scrofa_pig/sus_train_chromosome_2',
               'output_block_dims': (10*79, 5*79)},
}

pretraining_parameter_dict = {
    # trainer设置
    'total_epoch_number': 2,
    'step_per_epoch': 14_0000,
    'learning_rate': 0.0006,
    'weight_decay': 0.01,
    'batch_size': 50,
    'warmup_lr_epoch_number': 0.3,
    'constant_lr_epoch_number': 0.7,
    'cosine_annealing_lr_epoch_number': 1,
    'is_calculate_pcc_and_scc': False,
    'output_folder': '1_train',
    'pretrain_snapshot_path': None,
    'snapshot_path': None,
    'is_run_validation_at_first': False,
    # 模型设置
    'input_block_channel': [20, 190, 30],
    'input_block_kernel_size': [3, 9, 21],
    'input_block_additional_conv_layer': (9, 9),
    'input_block_exponential_activation': True,
    'width': [300, 360, 420, 480, 540, 600],
    'depth': [2, 2, 2, 2, 2, 2],
    'kernel_size': [9, 9, 9, 9, 9, 9],
    'stride': [4, 4, 4, 4, 4, 2],
    # dataset & dataloader设置
    'input_dna_length': 9*2**11,
    'train_num_workers': 2,
    'validation_num_workers': 4,
    'minimal_coverage': 5,
    'loss_weight_factor': 5,
    'max_loss_weight_factor': 1.0,
    'reverse_complement_augmentation': True,
    # 训练集文件设置
    'validation_set_file_name': '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/11_human1682_dataset/human1681_smooth_raw_window3_validation_chromosome.pkl',
    'genome_fasta_file_name': '/home/chenfaming/genome/ucsc_hg38/hg38.fa',
}

full_model_parameter_dict = {
    # trainer设置
    'total_epoch_number': 3,
    'max_step_per_epoch': 100_0000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0005,
    'weight_decay': 0.01,
    'batch_size': 50,
    'warmup_lr_epoch_number': 1,
    'constant_lr_epoch_number': 1,
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
    'keep_raw_methylation': True,
    'reverse_complement_augmentation': True,
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

human_encode_parameter_dict = {
    # trainer设置
    'total_epoch_number': 3,
    'max_step_per_epoch': 100_0000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0006,
    'weight_decay': 0.01,
    'batch_size': 150,
    'warmup_lr_epoch_number': 1,
    'constant_lr_epoch_number': 1,
    'output_folder': '10_finetune_human_encode',
    'pretrain_snapshot_path': '1_train/snapshot/snapshot_epoch_1.pth',
    'is_load_output_block_pretrain_weight': False,
    'snapshot_path': None,
    'is_run_validation_at_first': False,
    # 模型设置
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
    # 训练集文件设置
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

human_atlas_parameter_dict = {
    # trainer设置
    'total_epoch_number': 3,
    'max_step_per_epoch': 100_0000,
    'learning_rate': 0.0001,
    'output_block_learning_rate': 0.0006,
    'weight_decay': 0.01,
    'batch_size': 150,
    'warmup_lr_epoch_number': 1,
    'constant_lr_epoch_number': 1,
    'output_folder': '11_finetune_human_atlas',
    'pretrain_snapshot_path': '1_train/snapshot/snapshot_epoch_1.pth',
    'is_load_output_block_pretrain_weight': False,
    'snapshot_path': None,
    'is_run_validation_at_first': False,
    # 模型设置
    'input_block_channel': [20, 190, 30],
    'input_block_kernel_size': [3, 9, 21],
    'input_block_additional_conv_layer': (9, 9),
    'input_block_exponential_activation': True,
    'width': [300, 360, 420, 480, 540, 600],
    'depth': [2, 2, 2, 2, 2, 2],
    'kernel_size': [9, 9, 9, 9, 9, 9],
    'stride': [4, 4, 4, 4, 4, 2],
    'output_block_dims': (207 * 40, 207 * 20, 207 * 5),
    # dataset & dataloader设置
    'input_dna_length': 9 * 2**11,
    'train_num_workers': 2,
    'validation_num_workers': 4,
    'loss_weight_factor': 5,
    'max_loss_weight_factor': 1.0,
    'minimal_coverage': 5,
    'keep_raw_methylation': True,
    'reverse_complement_augmentation': True,
    # 训练集文件设置
    'train_set_file': '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
                      'human_atlas_smooth_raw_window3_train_chromosome.pkl',
    'bed_dataset_to_repetition_dict': {
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
        'human_atlas_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend500.uniq.bed': 1,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
        'human_atlas_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend250.uniq.bed': 1,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
        'human_atlas_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_extend100.uniq.bed': 1,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
        'human_atlas_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme.uniq.bed': 1,
        '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
        'human_atlas_smooth_raw_window3_train_chromosome_coordinate.inter_CTS_unme_train_repetition.uniq.bed': 1,
    },
    'validation_set_file': '/home/chenfaming/pool1/project/231009_DNA_methylation_data/241211_human_data/18_human_atlas_dataset/2_dataset_file/'
                           'human_atlas_smooth_raw_window3_validation_chromosome.pkl',
    'genome_fasta_file': '/home/chenfaming/genome/ucsc_hg38/hg38.fa',
}
