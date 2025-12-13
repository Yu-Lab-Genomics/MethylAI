import os
from abc import ABC, abstractmethod
import glob
import polars as pl
from MethylAI.src.utils.utils import check_output_folder

class MethylationFile(ABC):
    def __init__(self, input_folder: str, input_file_suffix: str, output_folder, output_log_file,
                 reference_cpg_coordinate_file: str = None, chr_list: list = None):
        # 文件相关设置
        self.input_folder = input_folder
        self.input_file_suffix = input_file_suffix
        if not self.input_file_suffix.startswith('.'):
            self.input_file_suffix = '.' + self.input_file_suffix
        self.methylation_file  = ''
        self.methylation_pldf = pl.DataFrame()
        # chr list，保留chr1-22，不保留XY和其他染色体或contig
        if not chr_list:
            self.chr_list = ['chr' + str(i) for i in range(1, 23)]
        else:
            self.chr_list = chr_list
        # 参考基因组CpG坐标
        self.reference_cpg_coordinate_file = reference_cpg_coordinate_file
        self.reference_cpg_coordinate_pldf = pl.DataFrame()
        if self.reference_cpg_coordinate_file:
            self._input_reference_cpg_coordinate()
        # self.process_log_list：每个文件产生一个str描述该文件存在的问题（目前有2个问题：坐标错误、染色体不全），每个文件运行结束后保存到log文件中
        self.process_log_list = []
        # 输出文件夹
        self.output_folder = output_folder
        check_output_folder(self.output_folder)
        self.output_log_file = f'{self.output_folder}/{output_log_file}'

    def _input_reference_cpg_coordinate(self):
        # 读取reference基因组的所有cg坐标
        col_name_list = ['chr', 'start', 'end']
        self.reference_cpg_coordinate_pldf = pl.read_csv(
            self.reference_cpg_coordinate_file,
            separator='\t',
            has_header=False,
            new_columns=col_name_list
        )
        # 保留chr1-22，不保留XY和其他染色体或contig
        self.reference_cpg_coordinate_pldf = self.reference_cpg_coordinate_pldf.filter(
            pl.col('chr').is_in(self.chr_list)
        )

    def preprocess_all_bed_file(self):
        methylation_file_list = glob.glob(f'{self.input_folder}/*{self.input_file_suffix}')
        for self.methylation_file in methylation_file_list:
            # 保存到log
            self.process_log_list.append(self.methylation_file)
            # 读取当前文件
            self._input_methylation_pldf()
            # 把数据合并到reference cpg coordinate，或仅合并正负链
            if self.reference_cpg_coordinate_file:
                self._merge_forward_reverse_reference_cpg_coordinate()
            else:
                self._merge_forward_reverse()
            self._output_methylation_pldf_and_log()

    @abstractmethod
    def _input_methylation_pldf(self):
        pass

    def _merge_forward_reverse(self):
        # 合并的on column
        merge_on_col_name_list = ['chr', 'start']
        # 最终需要的列 & 重命名
        final_col_list = ['chr', 'start', 'end', 'merge_mc', 'merge_cov']
        rename_final_col_list = ['chr', 'start', 'end', 'mc', 'cov']
        # 根据chr、start排序
        self.methylation_pldf = self.methylation_pldf.sort(by=['chr', 'start'])
        # 记录mc_sum、cov_sum
        self._record_mc_cov_info_to_log('before merge')
        # 拆分正负链
        forward_methylation_pldf = self.methylation_pldf.filter(
            pl.col('strand') == '+'
        )
        reverse_methylation_pldf = self.methylation_pldf.filter(
            pl.col('strand') == '-'
        )
        reverse_methylation_pldf = reverse_methylation_pldf.with_columns((pl.col('start') - 1))
        # 合并
        merge_methylation_pldf = forward_methylation_pldf.join(
            reverse_methylation_pldf, on=merge_on_col_name_list, how='full', coalesce=True
        )
        # 把null填0，然后正负链的值相加，计算end位置
        merge_methylation_pldf = merge_methylation_pldf.fill_null(pl.lit(0))
        merge_methylation_pldf = merge_methylation_pldf.with_columns(
            (pl.col('mc') + pl.col('mc_right')).alias('merge_mc'),
            (pl.col('cov') + pl.col('cov_right')).alias('merge_cov'),
            (pl.col('start') + 2).alias('end')
        )
        # 选择需要保留的列
        merge_methylation_pldf = merge_methylation_pldf[final_col_list]
        # 把merge好的dataframe赋值给self.methylation_pldf
        self.methylation_pldf = merge_methylation_pldf
        self.methylation_pldf.columns = rename_final_col_list
        # 记录mc_sum、cov_sum
        self._record_mc_cov_info_to_log('after merge')

    def _merge_forward_reverse_reference_cpg_coordinate(self):
        # 合并前需要保留的列
        select_col_list = ['chr', 'start', 'mc', 'cov']
        # 合并的on column
        join_on_col_list = ['chr', 'start']
        # 最终需要的列 & 重命名
        final_col_list = ['chr', 'start', 'end', 'merge_mc', 'merge_cov']
        rename_final_col_list = ['chr', 'start', 'end', 'mc', 'cov']
        # 根据chr、start排序
        self.methylation_pldf = self.methylation_pldf.sort(by=['chr', 'start'])
        # 记录mc_sum、cov_sum
        self._record_mc_cov_info_to_log('before merge')
        # 拆分正负链
        forward_methylation_pldf = self.reference_cpg_coordinate_pldf.join(
            self.methylation_pldf, on=join_on_col_list, how='left', coalesce=True
        )
        forward_methylation_pldf = forward_methylation_pldf[select_col_list]
        reverse_methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col('start') - 1).alias('start')
        )
        reverse_methylation_pldf = self.reference_cpg_coordinate_pldf.join(
            reverse_methylation_pldf, on=join_on_col_list, how='left', coalesce=True
        )
        reverse_methylation_pldf = reverse_methylation_pldf[select_col_list]
        # 合并
        merge_methylation_pldf = forward_methylation_pldf.join(
            reverse_methylation_pldf, on=join_on_col_list, how='full', coalesce=True
        )
        # 把null填0，然后正负链的值相加、计算end位置
        merge_methylation_pldf = merge_methylation_pldf.fill_null(pl.lit(0))
        merge_methylation_pldf = merge_methylation_pldf.with_columns(
            (pl.col('mc') + pl.col('mc_right')).alias('merge_mc'),
            (pl.col('cov') + pl.col('cov_right')).alias('merge_cov'),
            (pl.col('start') + 2).alias('end')
        )
        # 选择需要保留的列
        merge_methylation_pldf = merge_methylation_pldf[final_col_list]
        # 把merge好的dataframe赋值给self.methylation_pldf
        self.methylation_pldf = merge_methylation_pldf
        self.methylation_pldf.columns = rename_final_col_list
        # 记录mc_sum、cov_sum
        self._record_mc_cov_info_to_log('after merge')

    def _record_mc_cov_info_to_log(self, state: str):
        # 记录mc_sum、cov_sum
        mc_sum = self.methylation_pldf['mc'].sum()
        cov_sum = self.methylation_pldf['cov'].sum()
        mc_cov_info = f'{state}, mc: {mc_sum}, cov: {cov_sum}'
        self.process_log_list.append(mc_cov_info)

    def _output_methylation_pldf_and_log(self):
        # base name
        base_name = os.path.basename(self.methylation_file)
        base_name = base_name.replace(self.input_file_suffix, '.preprocessed.txt')
        # output name
        output_file = f'{self.output_folder}/{base_name}'
        print(f'output: {output_file}')
        self.methylation_pldf.write_csv(output_file, separator='\t')
        # 输出self.process_log_list到log文件，然后清空self.process_log_list
        output_process_log = '; '.join(self.process_log_list) + '\n'
        with open(self.output_log_file, 'a') as file:
            file.write(output_process_log)
        self.process_log_list = []


class EncodeMethylationFile(MethylationFile):
    def _input_methylation_pldf(self):
        # 列名
        # 处为ENCODE数据独有的问题，数据中存异常文件(ENCFF782JXT.bed.gz)，其他文件不一样
        col_11_name_list = ['chr', 'start', 'end', 'V4', 'V5', 'strand', 'V7', 'V8', 'V9', 'cov', 'mc_precent']
        col_name_list = ['chr', 'start', 'end', 'V4', 'V5', 'strand', 'V7', 'V8', 'V9', 'cov', 'mc_precent',
                         'ref_genotype', 'sample_genotype', 'quality_score_genotype']
        # 读取文件
        print(f'input: {self.methylation_file}')
        self.methylation_pldf = pl.read_csv(self.methylation_file, separator='\t', has_header=False)
        if self.methylation_pldf.shape[1] == 11:
            # 在log中进行记录
            log_str = 'col_11'
            print(log_str)
            self.process_log_list.append(log_str)
            # 设置列名
            self.methylation_pldf.columns = col_11_name_list
        else:
            # 设置列名
            self.methylation_pldf.columns = col_name_list
            # 保留type为CG的类型
            self.methylation_pldf = self.methylation_pldf.filter(
                pl.col('sample_genotype').is_in(['CG'])
            )
        # 保留chr_list中的chr
        self.methylation_pldf = self.methylation_pldf.filter(
            pl.col('chr').is_in(self.chr_list),
        )
        # 计算mc
        self.methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col('cov') * pl.col('mc_precent') / 100.0).alias('mc').round(0)
        )
        # 保留需要的列
        self.methylation_pldf = self.methylation_pldf[['chr', 'start', 'mc', 'cov']]
        # 把mc, start, end 设置为int类型
        self.methylation_pldf = self.methylation_pldf.cast({'mc': int, 'start': int})


class BismarkMethylationFile(MethylationFile):
    def _input_methylation_pldf(self):
        # 列名
        col_name_list = ['chr', 'start', 'end', 'mc_present', 'mc', 'unmc']
        # 读取文件
        print(f'input: {self.methylation_file}')
        self.methylation_pldf = pl.read_csv(self.methylation_file, separator='\t', has_header=False, infer_schema_length=100000)
        self.methylation_pldf.columns = col_name_list
        # 保留type为CG的类型，保留chr_list中的chr
        self.methylation_pldf = self.methylation_pldf.filter(
            pl.col('chr').is_in(self.chr_list),
        )
        # 计算cov
        self.methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col('mc') + pl.col('unmc')).alias('cov')
        )
        # 保留需要的列
        self.methylation_pldf = self.methylation_pldf[['chr', 'start', 'mc', 'cov']]
        # 把mc, start, end 设置为int类型
        self.methylation_pldf = self.methylation_pldf.cast({'mc': int, 'start': int})



