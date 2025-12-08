import polars as pl
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.utils import check_output_folder, debug_methods

@debug_methods
class LowMethylationRegion:
    def __init__(self, complete_dataset_file: str, output_folder: str, output_prefix: str):
        # 文件名
        self.complete_dataset_file = complete_dataset_file
        # 读入数据
        self.dataset_pldf = pl.DataFrame()
        self._input_data()
        # 正在处理中的数据
        self.processing_dataset_index = -1
        self.processing_dataset_pldf = pl.DataFrame()
        # 输出设置
        self.output_folder = output_folder
        check_output_folder(self.output_folder)
        self.output_prefix = f'{output_folder}/{output_prefix}'

    def _input_data(self):
        self.dataset_pldf = pl.read_csv(self.complete_dataset_file, separator='\t')

    def select_dataset_index(self, dataset_index: int):
        self.processing_dataset_index = dataset_index
        coordinate_col_list = self.dataset_pldf.columns[0: 3]
        col_postfix = f'_{self.processing_dataset_index}'
        select_col_list = coordinate_col_list + [col for col in self.dataset_pldf.columns if col.endswith(col_postfix)]
        self.processing_dataset_pldf = self.dataset_pldf[select_col_list]

    def mark_low_me_region(self, threshold_low_me_value=0.25, threshold_min_cpg_num=5, threshold_min_region_len=0):
        smooth_col = f'smooth_{self.processing_dataset_index}'
        self.processing_dataset_pldf = self.processing_dataset_pldf.with_columns(
            (pl.col(smooth_col) < threshold_low_me_value).alias('is_smooth_low_me')
        ).with_columns(
            ((pl.col('is_smooth_low_me') != pl.col('is_smooth_low_me').shift(1)) |
             (pl.col('chr') != pl.col('chr').shift(1))).fill_null(True).cum_sum().alias('region_id')
        ).with_columns(
            pl.len().over('region_id').alias('region_cg_num'),
            (pl.col('end').max().over('region_id') - pl.col('start').min().over('region_id')).alias('region_len')
        ).with_columns(
            pl.when(pl.col('is_smooth_low_me') &
                    (pl.col('region_cg_num') >= threshold_min_cpg_num) &
                    (pl.col('region_len') >= threshold_min_region_len))
            .then(pl.col('region_id'))
            .otherwise(-1)
            .alias('low_me_region_id')
        )

    def retain_low_me_region_and_mark_represent_cpg(self, interval=1000):
        self.processing_dataset_pldf = self.processing_dataset_pldf.filter(
            pl.col('low_me_region_id') >= 0
        )
        self.processing_dataset_pldf = self.processing_dataset_pldf.with_columns(
            pl.col('start').min().over('low_me_region_id').alias('low_me_region_start'),
            pl.col('end').max().over('low_me_region_id').alias('low_me_region_end'),
        )
        processing_dataset_pldf_list = []
        def get_mid_pos(interval_str):
            interval_str = interval_str.strip('[]()')
            pos_tuple = map(int, [x.strip() for x in interval_str.split(', ')])
            mid_pos = sum(pos_tuple) // 2
            return mid_pos

        for name, group_pldf in self.processing_dataset_pldf.group_by(['low_me_region_id']):
            # 确定低甲基化区域的长度
            region_start = group_pldf['start'].min()
            region_end = group_pldf['end'].max()
            region_length = region_end - region_start
            # 如长度>interval，把开始的窗口往左挪动，使得取样的CpG尽可能靠近中间
            if region_length > interval:
                shift_distance = (interval - (region_length % interval)) // 2
                shift_region_start = region_start - shift_distance
            else:
                shift_region_start = region_start
            # 生成采样区间（左闭右开）
            bins = list(range(shift_region_start, region_end, interval))
            bins[0] = region_start - 1
            bins.append(region_end)
            group_pldf = group_pldf.with_columns(
                pl.col('start').cut(bins, include_breaks=True, left_closed=True).alias('cut')
            ).unnest('cut').with_columns(
                pl.col('category').map_elements(get_mid_pos, return_dtype=pl.Int64).alias('mid_pos')
            ).with_columns(
                (pl.col('start') - pl.col('mid_pos')).abs().alias('abs(start-mid_pos)')
            ).with_columns(
                pl.lit(1).alias('for_cum_count')
            ).with_columns(
                ((pl.cum_count('for_cum_count') - 1).over(['breakpoint']) == pl.col('abs(start-mid_pos)').arg_min()
                 .over(['breakpoint'])).cast(pl.Int64).alias('represent_cpg')
            ).drop('for_cum_count')
            processing_dataset_pldf_list.append(group_pldf)
        self.processing_dataset_pldf = pl.concat(processing_dataset_pldf_list, how='vertical')
        self.processing_dataset_pldf = self.processing_dataset_pldf.sort(by=['chr', 'start'])

    def retain_representative_cpg(self):
        self.processing_dataset_pldf = self.processing_dataset_pldf.filter(
            pl.col('represent_cpg') == 1
        )

    def output_processing_dataset_pldf(self, output_file):
        output_file = f'{self.output_prefix}_{output_file}'
        print(f'output: {output_file}')
        self.processing_dataset_pldf.write_csv(output_file, separator='\t')

    def output_and_clean_processing_dataset_pldf(self, output_file):
        output_file = f'{self.output_prefix}_{output_file}'
        print(f'output: {output_file}')
        self.processing_dataset_pldf.write_csv(output_file, separator='\t')
        self.processing_dataset_pldf = pl.DataFrame()
