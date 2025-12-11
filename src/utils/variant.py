import polars as pl
from MethylAI.src.utils.utils import check_output_folder

class VariantAnalysis:
    def __init__(self, variant_effect_file: str, output_folder: str, output_prefix: str):
        self.variant_effect_file = variant_effect_file
        self.variant_effect_pldf = pl.DataFrame()
        self._input_data()
        # result
        self.variant_analysis_result_pldf = pl.DataFrame()
        self.variant_vcf_pldf = pl.DataFrame()
        self.output_folder = output_folder
        self.output_prefix = f'{self.output_folder}/{output_prefix}'
        check_output_folder(self.output_folder)

    def _input_data(self):
        self.variant_effect_pldf = pl.read_csv(self.variant_effect_file, separator='\t')
        self.variant_effect_pldf = self.variant_effect_pldf.filter(
            pl.col('cg_change') != 1
        )

    def generate_variant_analysis_result(
            self, is_low_methylation: bool, threshold_min_variant_effect: float, dataset_index: int,
            variant_active_motif_detail_file: str = None
    ):
        assert (threshold_min_variant_effect >= 0 if is_low_methylation
                else threshold_min_variant_effect <= 0)
        ref_col = f'ref_prediction_smooth_{dataset_index}'
        alt_col = f'alt_prediction_smooth_{dataset_index}'
        effect_col = f'effect_smooth_{dataset_index}'
        self.variant_effect_pldf = self.variant_effect_pldf.with_columns(
            (pl.col(alt_col) - pl.col(ref_col)).alias(effect_col)
        )
        if is_low_methylation:
            self.variant_effect_pldf = self.variant_effect_pldf.sort([effect_col], descending=[True])
            self.variant_analysis_result_pldf = self.variant_effect_pldf.group_by(['RSID', 'ALT_split']).head(1)
            self.variant_analysis_result_pldf = self.variant_analysis_result_pldf.filter(
                pl.col(effect_col) >= threshold_min_variant_effect
            )
        else:
            self.variant_effect_pldf = self.variant_effect_pldf.sort([effect_col], descending=[False])
            self.variant_analysis_result_pldf = self.variant_effect_pldf.group_by(['RSID', 'ALT_split']).head(1)
            self.variant_analysis_result_pldf = self.variant_analysis_result_pldf.filter(
                pl.col(effect_col) <= threshold_min_variant_effect
            )
        if variant_active_motif_detail_file:
            variant_active_motif_pldf = pl.read_csv(
                variant_active_motif_detail_file, separator='\t', has_header=False,
                new_columns=['chr', 'POS', 'RSID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO',
                             'motif_chr', 'motif_start', 'motif_end', 'motif_id_name']
            )
            variant_active_motif_pldf = variant_active_motif_pldf[['RSID', 'motif_start', 'motif_end', 'motif_id_name']].unique()
            self.variant_analysis_result_pldf = self.variant_analysis_result_pldf.join(
                variant_active_motif_pldf, how='left', on=['RSID'], coalesce=True
            )
            self.variant_analysis_result_pldf = self.variant_analysis_result_pldf.sort(
                ['chr', 'POS', 'motif_start', 'motif_id_name'])
        else:
            self.variant_analysis_result_pldf = self.variant_analysis_result_pldf.sort(
                ['chr', 'POS'])

    def output_result(self):
        variant_analysis_result_file = f'{self.output_prefix}_variant_analysis_result.txt'
        self.variant_analysis_result_pldf.write_csv(variant_analysis_result_file, separator='\t')


























