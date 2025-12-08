import polars as pl
import numpy as np
import datetime
import re
import os
import subprocess
from typing import Sequence
from MethylAI.src.utils.utils import check_output_folder, debug_methods


class CaptumResult:
    def __init__(self, captum_cpg_file, dna_attribution_file):
        # 输入文件名
        self.captum_cpg_file = captum_cpg_file
        self.dna_attribution_file = dna_attribution_file
        # 输入文件内容
        self.captum_cpg_pldf = pl.DataFrame()
        self.dna_attribution_numpy = np.array(0)
        # 读取上述文件
        self._input_file()

    def _input_file(self):
        print(f'input captum cpg file: {self.captum_cpg_file}\n'
              f'input dna attribution file: {self.dna_attribution_file}')
        self.captum_cpg_pldf = pl.read_csv(self.captum_cpg_file, separator='\t')
        self.dna_attribution_numpy = np.load(self.dna_attribution_file)
        # 断言长度相同
        assert len(self.captum_cpg_pldf) == len(self.dna_attribution_numpy)

    def __len__(self):
        return len(self.captum_cpg_pldf)

    def get_dna_coordinate(self, idx):
        input_dna_chr = self.captum_cpg_pldf[idx, 'chr']
        input_dna_start = self.captum_cpg_pldf[idx, 'start']
        input_dna_end = self.captum_cpg_pldf[idx, 'end']
        return input_dna_chr, input_dna_start, input_dna_end

    def get_cg_coordinate(self, idx):
        cg_chr = self.captum_cpg_pldf[idx, 'chr']
        cg_start = self.captum_cpg_pldf[idx, 'start']
        cg_end = self.captum_cpg_pldf[idx, 'end']
        return cg_chr, cg_start, cg_end

    def get_all_coordinate(self, idx):
        input_dna_chr = self.captum_cpg_pldf[idx, 'chr']
        input_dna_start = self.captum_cpg_pldf[idx, 'input_dna_start']
        input_dna_end = self.captum_cpg_pldf[idx, 'input_dna_end']
        cg_start = self.captum_cpg_pldf[idx, 'start']
        cg_end = self.captum_cpg_pldf[idx, 'end']
        return input_dna_chr, input_dna_start, input_dna_end, cg_start, cg_end

    def get_attribution(self, seq_index, start, end):
        return self.dna_attribution_numpy[seq_index, start: end]

    def get_attribution_shape(self):
        return self.dna_attribution_numpy.shape


class JasparBed:
    def __init__(self, jaspar_bed_file):
        # 文件名
        self.jaspar_bed_file = jaspar_bed_file
        # dataframe
        self.jaspar_bed_pldf: pl.DataFrame
        self._input_file()

    def _input_file(self):
        print(f'input: {self.jaspar_bed_file}')
        # 读入文件并设置表头
        self.jaspar_bed_pldf = pl.read_csv(self.jaspar_bed_file, has_header=False, separator='\t')
        self.jaspar_bed_pldf.columns = ['chr', 'motif_start', 'motif_end', 'motif_id', 'motif_score', 'motif_strand', 'motif_name']

    def get_dataframe_by_coordinate(self, input_dna_chr, input_dna_start, input_dna_end, cg_start, cg_end):
        input_dna_jaspar_bed_pldf = self.jaspar_bed_pldf.filter(
            (pl.col('chr') == input_dna_chr) & (pl.col('motif_start') > input_dna_start) & (pl.col('motif_end') < input_dna_end)
        )
        # 计算motif长度、motif相对坐标、motif与cg的距离
        input_dna_jaspar_bed_pldf = input_dna_jaspar_bed_pldf.with_columns(
            (pl.col('motif_end') - pl.col('motif_start')).alias('motif_len'),
            (pl.col('motif_start') - input_dna_start).alias('motif_relative_start'),
            (pl.col('motif_end') - input_dna_start).alias('motif_relative_end'),
            pl.when(pl.col('motif_end') < cg_start).then(pl.col('motif_end') - cg_start)
            .when(pl.col('motif_start') > cg_end).then(pl.col('motif_start') - cg_end)
            .otherwise(0)
            .alias('motif_cg_distance')
        )
        return input_dna_jaspar_bed_pldf


class MotifStatistic:
    def __init__(self, captum_result: CaptumResult, jaspar_bed: JasparBed, print_per_step, output_folder, output_prefix):
        # 前面构建好的2个类
        self.captum_result = captum_result
        self.jaspar_bed = jaspar_bed
        # 统计结果
        self.motif_statistic_list = []
        self.motif_statistic_pldf = pl.DataFrame()
        # print设置
        self.print_per_step = print_per_step
        # 设置输出文件
        self.output_folder = output_folder
        self.output_prefix = os.path.join(self.output_folder, output_prefix)
        check_output_folder(self.output_folder)

    def do_motif_statistic(self):
        start_time = datetime.datetime.now()
        captum_result_len = len(self.captum_result)
        for seq_index in range(captum_result_len):
            input_dna_chr, input_dna_start, input_dna_end, cg_start, cg_end = self.captum_result.get_all_coordinate(seq_index)
            input_dna_jaspar_bed_pldf = self.jaspar_bed.get_dataframe_by_coordinate(
                input_dna_chr, input_dna_start, input_dna_end, cg_start, cg_end
            )
            motif_attribution_sum_list = []
            motif_attribution_abs_sum_list = []
            motif_attribution_str_list = []
            for bed_row in input_dna_jaspar_bed_pldf.iter_rows(named=True):
                motif_relative_start = bed_row['motif_relative_start']
                motif_relative_end = bed_row['motif_relative_end']
                motif_attribution_numpy = self.captum_result.get_attribution(seq_index, motif_relative_start, motif_relative_end)
                motif_attribution_sum = np.sum(motif_attribution_numpy)
                motif_attribution_abs_sum = np.sum(np.abs(motif_attribution_numpy))
                motif_attribution_str = ','.join(map(lambda x: format(x, '.4g'), motif_attribution_numpy.tolist()))
                # 保存至list
                motif_attribution_sum_list.append(motif_attribution_sum)
                motif_attribution_abs_sum_list.append(motif_attribution_abs_sum)
                motif_attribution_str_list.append(motif_attribution_str)
            # 第2个for循环结束
            input_dna_jaspar_bed_pldf = input_dna_jaspar_bed_pldf.with_columns(
                # 加入cg坐标信息
                pl.lit(input_dna_start).alias('input_dna_start'),
                pl.lit(input_dna_end).alias('input_dna_end'),
                pl.lit(cg_start).alias('cg_start'),
                pl.lit(cg_end).alias('cg_end'),
                # 加入attribution_sum信息
                (pl.Series(motif_attribution_sum_list)).alias('motif_attribution_sum'),
                (pl.Series(motif_attribution_abs_sum_list)).alias('motif_attribution_abs_sum'),
                (pl.Series(motif_attribution_str_list)).alias('motif_attribution_str'),
            ).with_columns(
                # 加入attribution_mean信息
                (pl.col('motif_attribution_sum') / pl.col('motif_len')).alias('motif_attribution_mean'),
                (pl.col('motif_attribution_abs_sum') / pl.col('motif_len')).alias('motif_attribution_abs_mean')
            )
            self.motif_statistic_list.append(input_dna_jaspar_bed_pldf)
            if seq_index % self.print_per_step == 0:
                print(f'{seq_index:5d}|{captum_result_len:5d}')
                using_time = datetime.datetime.now() - start_time
                print(f'using_time: {using_time}\n')
        # for循环结束，拼接self.motif_statistic_pldf
        self.motif_statistic_pldf = pl.concat(self.motif_statistic_list)

    def output_dataframe(self):
        output_file = f'{self.output_prefix}_motif_statistic_dataframe.txt'
        print(f'output: {output_file}')
        self.motif_statistic_pldf.write_csv(output_file, separator='\t')

class EvaluationResult:
    def __init__(self, evaluation_file: str, output_folder: str):
        # 文件名 & 文件
        self.evaluation_file = evaluation_file
        self.evaluation_pldf = pl.DataFrame()
        # col_list
        self.true_col_list = []
        self.prediction_col_list = []
        self.coverage_col_list = []
        self._input_file()
        # col_name
        self.true_col = ''
        self.prediction_col = ''
        self.coverage_col = ''
        self.abs_diff_col = ''
        # select_pldf
        self.selected_evaluation_pldf = pl.DataFrame()
        # output setting
        self.output_folder = output_folder

    def _input_file(self):
        print('_input_file')
        self.evaluation_pldf = pl.read_csv(self.evaluation_file, separator='\t', has_header=True)

    def infer_col_name_1(self, true_col_prefix_tuple: tuple, model_output_index: int):
        # 读取true_col_list和prediction_col_list
        self.true_col_list = [col_name for col_name in self.evaluation_pldf.columns if
                              col_name.startswith(true_col_prefix_tuple)]
        self.prediction_col_list = [col_name for col_name in self.evaluation_pldf.columns if
                                    col_name.startswith('prediction')]
        self.coverage_col_list = [col_name for col_name in self.evaluation_pldf.columns if
                                  col_name.startswith('coverage')]
        self.true_col = self.true_col_list[model_output_index]
        self.prediction_col = self.prediction_col_list[model_output_index]
        self.coverage_col = self.coverage_col_list[model_output_index]
        self.abs_diff_col = f'abs_diff_{self.true_col}'

    def infer_col_name_2(self, dataset_index: int):
        self.true_col = f'smooth_{dataset_index}'
        self.prediction_col = f'prediction_smooth_{dataset_index}'
        self.coverage_col = f'coverage_{dataset_index}'
        self.abs_diff_col = f'abs_diff_{self.true_col}'

    def generate_selected_evaluation_pldf(self):
        print(f'true_col: {self.true_col}, prediction_col: {self.prediction_col}, coverage_col: {self.coverage_col}, '
              f'abs_diff_col: {self.abs_diff_col}')
        self.selected_evaluation_pldf = self.evaluation_pldf.with_columns(
            pl.concat_str([pl.col('chr'), pl.col('start')], separator='_').alias('cg_chr_start'),
            (pl.col(self.true_col) - pl.col(self.prediction_col)).abs().alias(self.abs_diff_col)
        )
        self.selected_evaluation_pldf = self.selected_evaluation_pldf[[
            'cg_chr_start', self.true_col, self.prediction_col, self.coverage_col, self.abs_diff_col
        ]]
        # 给列重命名
        col_rename_dict = {
            self.true_col: 'smooth',
            self.prediction_col: 'prediction_smooth',
            self.coverage_col: 'coverage',
            self.abs_diff_col: 'abs_diff_smooth',
        }
        self.selected_evaluation_pldf = self.selected_evaluation_pldf.rename(col_rename_dict)
        self.selected_evaluation_pldf = self.selected_evaluation_pldf.with_columns(
            pl.lit(self.true_col).alias('smooth_index'),
        )

    def get_selected_evaluation_pldf(self):
        return self.selected_evaluation_pldf

    def output_selected_evaluation_pldf(self, output_file):
        output_file = f'{self.output_folder}/{output_file}'
        print(f'output: {output_file}')
        self.selected_evaluation_pldf.write_csv(output_file, separator='\t')


class MotifAnalysis:
    def __init__(self, motif_statistic_file: str, output_folder: str, output_prefix: str):
        # 文件名
        self.motif_statistic_file = motif_statistic_file
        self.motif_statistic_pldf = pl.DataFrame()
        self.captum_cpg_pldf = pl.DataFrame()
        self._input_file()
        # 记录
        self.is_join_evaluation_result = False
        self.captum_cpg_df_region_col = ''
        # 结果
        self.motif_statistic_filtered_pldf = pl.DataFrame()
        self.active_motif_statistic_pldf = pl.DataFrame()
        self.active_motif_summary_pldf = pl.DataFrame()
        self.all_motif_bed_pldf = pl.DataFrame()
        self.active_motif_bed_pldf = pl.DataFrame()
        self.output_folder = output_folder
        self.output_prefix = f'{self.output_folder}/{output_prefix}'
        check_output_folder(self.output_folder)

    def _input_file(self):
        # 读入文件
        self.motif_statistic_pldf = pl.read_csv(self.motif_statistic_file, separator='\t')
        # 拼接cg_chr_start作为cg_id；拼接motif_id_name；计算motif_activation_score
        self.motif_statistic_pldf = self.motif_statistic_pldf.drop(['motif_attribution_str'])
        self.motif_statistic_pldf = self.motif_statistic_pldf.with_columns(
            pl.concat_str([pl.col('chr'), pl.col('cg_start')], separator='_').alias('cg_chr_start'),
            pl.concat_str([pl.col('motif_id'), pl.col('motif_name')], separator='_').alias('motif_id_name'),
            (pl.col('motif_attribution_mean') * pl.col('motif_score')).alias('motif_activation_score')
        )
        # 以下操作使在同一位置的motif只保留1个motif_score最大的；同一坐标按照motif_score排序
        self.motif_statistic_pldf = self.motif_statistic_pldf.sort(
            ['motif_score'], descending=[True]
        )
        self.motif_statistic_pldf = self.motif_statistic_pldf.group_by(
            ['cg_chr_start', 'motif_start', 'motif_end', 'motif_id']
        ).head(1)

    def join_captum_cpg_pldf_region_col(self, captum_cpg_file, region_col: str):
        self.captum_cpg_df_region_col = region_col
        self.captum_cpg_pldf = pl.read_csv(captum_cpg_file, separator='\t')
        self.captum_cpg_pldf = self.captum_cpg_pldf.with_columns(
            pl.concat_str([pl.col('chr'), pl.col('start')], separator='_').alias('cg_chr_start'),
        )
        self.captum_cpg_pldf = self.captum_cpg_pldf[['cg_chr_start', region_col]].unique()
        self.motif_statistic_pldf = self.motif_statistic_pldf.join(
            self.captum_cpg_pldf, how='left', on=['cg_chr_start'], coalesce=True
        )

    def join_evaluation_pldf(self, evaluation_file: str, dataset_index: int):
        self.is_join_evaluation_result = True
        evaluation_result = EvaluationResult(
            evaluation_file=evaluation_file,
            output_folder=self.output_folder,
        )
        evaluation_result.infer_col_name_2(dataset_index=dataset_index)
        evaluation_result.generate_selected_evaluation_pldf()
        evaluation_result.output_selected_evaluation_pldf('evaluation_dataframe.txt')
        evaluation_pldf = evaluation_result.get_selected_evaluation_pldf()
        self.motif_statistic_pldf = self.motif_statistic_pldf.join(
            evaluation_pldf, on='cg_chr_start', how='left', coalesce=True
        )

    def filter_motif_statistic_pldf(
            self, threshold_max_motif_cpg_distance: int,  threshold_max_prediction_error: float = None
    ):
        # filter
        if threshold_max_prediction_error:
            self.motif_statistic_filtered_pldf = self.motif_statistic_pldf.filter(
                ((pl.col('motif_cg_distance').abs() < threshold_max_motif_cpg_distance) &
                 (pl.col('abs_diff_smooth') <= threshold_max_prediction_error))
            )
        else:
            self.motif_statistic_filtered_pldf = self.motif_statistic_pldf.filter(
                (pl.col('motif_cg_distance').abs() < threshold_max_motif_cpg_distance)
            )

    def set_motif_statistic_filtered_pldf_col_order(self):
        if self.is_join_evaluation_result and self.captum_cpg_df_region_col:
            self.motif_statistic_pldf = self.motif_statistic_pldf[[
                'chr', 'cg_chr_start', self.captum_cpg_df_region_col, 'cg_start', 'cg_end', 'input_dna_start', 'input_dna_end',
                'smooth', 'prediction_smooth', 'coverage', 'abs_diff_smooth', 'smooth_index',
                'motif_id_name', 'motif_id', 'motif_name', 'motif_start', 'motif_end', 'motif_len', 'motif_strand',
                'motif_relative_start', 'motif_relative_end', 'motif_cg_distance',
                'motif_score', 'motif_attribution_sum', 'motif_attribution_abs_sum',
                'motif_attribution_mean', 'motif_attribution_abs_mean', 'motif_activation_score'
            ]]
        elif self.is_join_evaluation_result:
            self.motif_statistic_pldf = self.motif_statistic_pldf[[
                'chr', 'cg_chr_start', 'cg_start', 'cg_end', 'input_dna_start', 'input_dna_end',
                'smooth', 'prediction_smooth', 'coverage', 'abs_diff_smooth', 'smooth_index',
                'motif_id_name', 'motif_id', 'motif_name', 'motif_start', 'motif_end', 'motif_len', 'motif_strand',
                'motif_relative_start', 'motif_relative_end', 'motif_cg_distance',
                'motif_score', 'motif_attribution_sum', 'motif_attribution_abs_sum',
                'motif_attribution_mean', 'motif_attribution_abs_mean', 'motif_activation_score'
            ]]
        elif self.captum_cpg_df_region_col:
            self.motif_statistic_pldf = self.motif_statistic_pldf[[
                'chr', 'cg_chr_start', self.captum_cpg_df_region_col, 'cg_start', 'cg_end', 'input_dna_start', 'input_dna_end',
                'smooth', 'prediction_smooth', 'coverage', 'abs_diff_smooth', 'smooth_index',
                'motif_id_name', 'motif_id', 'motif_name', 'motif_start', 'motif_end', 'motif_len', 'motif_strand',
                'motif_relative_start', 'motif_relative_end', 'motif_cg_distance',
                'motif_score', 'motif_attribution_sum', 'motif_attribution_abs_sum',
                'motif_attribution_mean', 'motif_attribution_abs_mean', 'motif_activation_score'
            ]]
        else:
            self.motif_statistic_pldf = self.motif_statistic_pldf[[
                'chr', 'cg_chr_start', 'cg_start', 'cg_end', 'input_dna_start', 'input_dna_end',
                'motif_id_name', 'motif_id', 'motif_name', 'motif_start', 'motif_end', 'motif_len', 'motif_strand',
                'motif_relative_start', 'motif_relative_end', 'motif_cg_distance',
                'motif_score', 'motif_attribution_sum', 'motif_attribution_abs_sum',
                'motif_attribution_mean', 'motif_attribution_abs_mean', 'motif_activation_score'
            ]]

    def generate_active_motif_pldf(self, is_low_methylation: bool, threshold_motif_attribution_mean,):
        assert (threshold_motif_attribution_mean < 0 if is_low_methylation
                else threshold_motif_attribution_mean > 0)
        # if-else筛选active motif
        if is_low_methylation:
            self.active_motif_statistic_pldf = self.motif_statistic_filtered_pldf.filter(
                pl.col('motif_attribution_mean') < threshold_motif_attribution_mean
            )
        else:
            self.active_motif_statistic_pldf = self.motif_statistic_filtered_pldf.filter(
                pl.col('motif_attribution_mean') > threshold_motif_attribution_mean
            )

    def generate_active_motif_summary_pldf(self):
        # 分别统计motif_activation_score
        self.active_motif_summary_pldf = self.active_motif_statistic_pldf.group_by('motif_id_name').agg(
            pl.col('motif_activation_score').sum().alias('motif_activation_score'),
            pl.col('motif_activation_score').mean().alias('motif_activation_score_mean'),
            pl.col('cg_chr_start').n_unique().alias('motif_regulated_window_number'),
            pl.col(self.captum_cpg_df_region_col).n_unique().alias('motif_regulated_region_number'),
        )
        motif_active_number_pldf = self.active_motif_statistic_pldf[[
            'chr', 'motif_start', 'motif_end', 'motif_id_name'
        ]].unique()
        motif_active_number_pldf = motif_active_number_pldf.group_by(['motif_id_name']).agg(
            pl.len().alias('motif_active_number')
        )
        motif_total_number_pldf = self.motif_statistic_filtered_pldf[[
            'chr', 'motif_start', 'motif_end', 'motif_id_name'
        ]].unique()
        motif_total_number_pldf = motif_total_number_pldf.group_by('motif_id_name').agg(
            pl.len().alias('motif_total_number')
        )
        # 合并数据框，排序
        self.active_motif_summary_pldf = self.active_motif_summary_pldf.join(
            motif_active_number_pldf, how='left', on=['motif_id_name']
        )
        self.active_motif_summary_pldf = self.active_motif_summary_pldf.join(
            motif_total_number_pldf, how='left', on=['motif_id_name']
        )
        self.active_motif_summary_pldf = self.active_motif_summary_pldf.with_columns(
            (pl.col('motif_active_number') / pl.col('motif_total_number')).alias('motif_active_ratio'),
            (pl.lit(self.motif_statistic_filtered_pldf['cg_chr_start'].n_unique())).alias('window_total_number'),
            (pl.lit(self.active_motif_statistic_pldf['cg_chr_start'].n_unique())).alias('window_with_active_motif_number'),
            (pl.lit(self.motif_statistic_filtered_pldf[self.captum_cpg_df_region_col].n_unique())).alias('region_total_number'),
            (pl.lit(self.active_motif_statistic_pldf[self.captum_cpg_df_region_col].n_unique())).alias('region_with_active_motif_number'),
        ).with_columns(
            (pl.col('motif_regulated_region_number') / pl.col('region_total_number')).alias('motif_regulated_region_ratio')
        )
        self.active_motif_summary_pldf = self.active_motif_summary_pldf.sort(
            'motif_regulated_region_number', descending=True
        )

    def generate_motif_bed_pldf(self):
        print('generate_motif_bed_pldf')
        self.all_motif_bed_pldf = self.motif_statistic_filtered_pldf[['chr', 'motif_start', 'motif_end', 'motif_id_name']]
        self.all_motif_bed_pldf = self.all_motif_bed_pldf.unique()
        self.all_motif_bed_pldf = self.all_motif_bed_pldf.sort(
            ['chr', 'motif_start', 'motif_id_name'], descending=[False, False, False]
        )
        self.active_motif_bed_pldf = self.active_motif_statistic_pldf[['chr', 'motif_start', 'motif_end', 'motif_id_name']]
        self.active_motif_bed_pldf = self.active_motif_bed_pldf.unique()
        self.active_motif_bed_pldf = self.active_motif_bed_pldf.sort(
            ['chr', 'motif_start', 'motif_id_name'], descending=[False, False, False]
        )

    def output_result(self, bedtools_path: str = None):
        # 文件名
        motif_statistic_file = f'{self.output_prefix}_motif_statistic.txt'
        motif_statistic_filtered_file = f'{self.output_prefix}_motif_statistic_filtered.txt'
        active_motif_statistic_file = f'{self.output_prefix}_active_motif_statistic.txt'
        active_motif_summary_file = f'{self.output_prefix}_active_motif_summary.txt'
        all_motif_bed_file = f'{self.output_prefix}_all_motif.bed'
        active_motif_bed_file = f'{self.output_prefix}_active_motif.bed'
        inactive_motif_bed_file = f'{self.output_prefix}_inactive_motif.bed'
        # 输出文件
        self.motif_statistic_pldf.write_csv(motif_statistic_file, separator='\t')
        self.motif_statistic_filtered_pldf.write_csv(motif_statistic_filtered_file, separator='\t')
        self.active_motif_statistic_pldf.write_csv(active_motif_statistic_file, separator='\t')
        self.active_motif_summary_pldf.write_csv(active_motif_summary_file, separator='\t')
        self.all_motif_bed_pldf.write_csv(all_motif_bed_file, separator='\t', include_header=False)
        self.active_motif_bed_pldf.write_csv(active_motif_bed_file, separator='\t', include_header=False)
        if bedtools_path:
            inactive_motif_command = f'{bedtools_path} intersect -v -a {all_motif_bed_file} -b {active_motif_bed_file} > {inactive_motif_bed_file}'
            print(inactive_motif_command)
            subprocess.run(inactive_motif_command, shell=True)

