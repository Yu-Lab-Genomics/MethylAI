import os
import polars as pl
import glob

class MethylationBedFile:
    def __init__(self, bed_file_pattern, hg38_cpg_coordinate_file, output_folder, output_log_file,
                 need_merge_forward_reverse: bool, chr_list: list = None,):
        # bed文件相关设置
        self.bed_file_pattern = bed_file_pattern
        self.bed_file_list = []
        self.running_bed_file = ''
        self.methylation_pldf = pl.DataFrame()
        # 设置：是否需要合并正负链
        self.need_merge_forward_reverse = need_merge_forward_reverse
        # chr list，保留chr1-22，不保留XY和其他染色体或contig
        if not chr_list:
            self.chr_list = ['chr' + str(i) for i in range(1, 23)]
        else:
            self.chr_list = chr_list
        # hg38基因组CpG坐标
        self.hg38_cpg_coordinate_file = hg38_cpg_coordinate_file
        self.hg38_cpg_coordinate_pldf = pl.DataFrame()
        self.input_cg_hg38_bed_file()
        # self.process_log_list：每个文件产生一个str描述该文件存在的问题（目前有2个问题：坐标错误、染色体不全），每个文件运行结束后保存到log文件中
        self.process_log_list = []
        # 输出文件夹
        self.output_folder = output_folder
        self.log_file_name = output_log_file

    def input_cg_hg38_bed_file(self):
        # 读取hg38基因组的所有cg坐标
        col_name_list = ['chr', 'start', 'end']
        self.hg38_cpg_coordinate_pldf = pl.read_csv(self.hg38_cpg_coordinate_file, separator='\t', has_header=False,
                                                  new_columns=col_name_list, n_threads=4)
        # 保留chr1-22，不保留XY和其他染色体或contig
        self.hg38_cpg_coordinate_pldf = self.hg38_cpg_coordinate_pldf.filter(pl.col('chr').is_in(self.chr_list))

    def preprocess_all_bed_file(self):
        self.bed_file_list = glob.glob(self.bed_file_pattern)
        for bed_file_name in self.bed_file_list:
            # print当前文件名，保存到log
            print(bed_file_name)
            self.process_log_list.append(bed_file_name)
            # 读取当前文件
            self.input_methylation_pldf(bed_file_name)
            # 如果需要则合并正负链
            if self.need_merge_forward_reverse:
                self.merge_forward_reverse()
            # 把数据合并到cg_hg38
            self.merge_hg38_cpg_coordinate()
            base_name = os.path.basename(bed_file_name)
            base_name = base_name.replace('.bed.gz', '.preprocessed.txt')
            output_file_name = os.path.join(self.output_folder, base_name)
            self.output_methylation_pldf_and_log(output_file_name)

    def input_methylation_pldf(self, bed_file_name):
        print('input_methylation_pldf')
        # 列名
        # 处为ENCODE数据独有的问题，数据中存异常文件(ENCFF782JXT.bed.gz)，其他文件不一样
        col_11_name_list = ['chr', 'start', 'end', 'V4', 'V5', 'strand', 'V7', 'V8', 'V9', 'cov', 'mc_precent']
        col_name_list = ['chr', 'start', 'end', 'V4', 'V5', 'strand', 'V7', 'V8', 'V9', 'cov', 'mc_precent',
                         'ref_genotype', 'sample_genotype', 'quality_score_genotype']
        # 读取文件
        self.methylation_pldf = pl.read_csv(bed_file_name, separator='\t', has_header=False, n_threads=48)
        #
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
            self.methylation_pldf = self.methylation_pldf.filter(pl.col('sample_genotype').is_in(['CG']))
        # 保留chr1-22，不保留XY和其他染色体或contig
        self.methylation_pldf = self.methylation_pldf.filter(pl.col('chr').is_in(self.chr_list))
        # 把start,end 设置为int类型
        self.methylation_pldf = self.methylation_pldf.cast({'start': int, 'end': int})
        # 根据chr、start排序
        self.methylation_pldf = self.methylation_pldf.sort(by=['chr', 'start'])
        # 计算mc
        self.methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col('cov') * pl.col('mc_precent') / 100.0).alias('mc').round(0)
        )
        self.methylation_pldf = self.methylation_pldf.cast({'mc': int})

    def merge_forward_reverse(self):
        print('merge_forward_reverse')
        # 需要合并的列
        select_col_name_list = ['chr', 'start', 'cov', 'mc']
        # 合并的on column
        merge_on_col_name_list = ['chr', 'start']
        # 拆分正负链
        forward_methylation_pldf = self.methylation_pldf.filter(pl.col('strand') == '+').select(select_col_name_list)
        reverse_methylation_pldf = self.methylation_pldf.filter(pl.col('strand') == '-').select(select_col_name_list)
        reverse_methylation_pldf = reverse_methylation_pldf.with_columns((pl.col('start') - 1))
        # 合并
        merge_methylation_pldf = forward_methylation_pldf.join(reverse_methylation_pldf, on=merge_on_col_name_list, how='full', coalesce=True)
        # 把null填0，然后正负链的值相加
        merge_methylation_pldf = merge_methylation_pldf.fill_null(pl.lit(0))
        merge_methylation_pldf = merge_methylation_pldf.with_columns((pl.col('mc') + pl.col('mc_right')).alias('merge_mc'))
        merge_methylation_pldf = merge_methylation_pldf.with_columns((pl.col('cov') + pl.col('cov_right')).alias('merge_cov'))
        # 计算end位置
        merge_methylation_pldf = merge_methylation_pldf.with_columns((pl.col('start') + 2).alias('end'))
        # 选择需要保留的列
        merge_methylation_pldf = merge_methylation_pldf[['chr', 'start', 'end', 'merge_mc', 'merge_cov']]
        # 把merge好的dataframe赋值给self.methylation_pldf
        self.methylation_pldf = merge_methylation_pldf
        self.methylation_pldf.columns = ['chr', 'start', 'end', 'mc', 'cov']

    def merge_hg38_cpg_coordinate(self):
        print('merge_cg_hg38_bed')
        merge_on_col_name_list = ['chr', 'start', 'end']
        merge_hg38_methylation_pldf = self.hg38_cpg_coordinate_pldf.join(self.methylation_pldf, on=merge_on_col_name_list, how='left', coalesce=True)
        # 缺失值填0，后续使用R能处理掉该缺失值
        merge_hg38_methylation_pldf = merge_hg38_methylation_pldf.fill_null(0)
        self.methylation_pldf = merge_hg38_methylation_pldf

    def output_methylation_pldf_and_log(self, output_file_name):
        print(f'output: {output_file_name}\n')
        self.methylation_pldf.write_csv(output_file_name, separator='\t', include_header=True)
        # 输出self.process_log_list到log文件，然后清空self.process_log_list
        output_process_log = '\t'.join(self.process_log_list) + '\n'
        with open(self.log_file_name, 'a') as file:
            file.write(output_process_log)
        self.process_log_list = []


def main_preprocess_methylation_bed_data():
    os.chdir('/home/chenfaming/pool2/project/231009_DNA_methylation_data/240701_ENCODE_Roadmap_data')
    methylation_bed_file = MethylationBedFile(
        bed_file_pattern='2_ENCODE_Roadmap_raw/*.bed.gz',
        hg38_cpg_coordinate_file='/home/chenfaming/pool2/project/231009_DNA_methylation_data/240709_GSE186458_human_methylome_atlas/1_test/CpG.hg38.bed',
        need_merge_forward_reverse=True,
        output_folder='8_preprocess',
        output_log_file='log_8_preprocess.log'
    )
    methylation_bed_file.preprocess_all_bed_file()

def main_command_line():
    import argparse
    parser = argparse.ArgumentParser(description='preprocess ENCODE DNA methylation file')
    parser.add_argument('input_file', help='输入文件路径')



if __name__ == "__main__":
    # main_preprocess_methylation_bed_data()
    main_command_line()







