import argparse
import polars as pl
import pandas as pd

def process_variant_cpg_file(variant_cpg_file: str):
    variant_cpg_pldf = pl.read_csv(
        variant_cpg_file, separator='\t', has_header=False,
        new_columns=['chr', 'POS', 'RSID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'cg_chr', 'cg_start', 'cg_end']
    )
    variant_cpg_pldf = variant_cpg_pldf.with_columns(
        pl.col('ALT').str.split(',').alias('ALT_split')
    ).explode('ALT_split').with_columns(
        (pl.col('POS') - 1).alias('variant_start'),
        pl.col('REF').str.len_chars().alias('variant_ref_len'),
        pl.col('ALT_split').str.len_chars().alias('variant_alt_len'),
    ).with_columns(
        pl.when(pl.col('variant_start') < pl.col('cg_start'))
        .then(pl.col('variant_start') - pl.col('cg_start'))
        .otherwise(pl.col('variant_start') - pl.col('cg_end') + pl.col('variant_ref_len'))
        .alias('variant_cg_distance')
    )
    variant_cpg_pldf = variant_cpg_pldf[[
        'chr', 'POS', 'RSID', 'REF', 'ALT', 'ALT_split', 'variant_start', 'variant_ref_len', 'variant_alt_len',
        'variant_cg_distance', 'cg_start', 'cg_end',
    ]]
    return variant_cpg_pldf

def output_variant_cpg_dataset_pldf(variant_cpg_dataset_pldf:pl.DataFrame, output_file: str):
    print(f'output: {output_file}')
    if output_file.endswith('pkl'):
        variant_cpg_dataset_df: pd.DataFrame = variant_cpg_dataset_pldf.to_pandas()
        variant_cpg_dataset_df.to_pickle(output_file)
    else:
        variant_cpg_dataset_pldf.write_csv(output_file, separator='\t')

def main_argparse():
    parser = argparse.ArgumentParser(description='Get variant CpG dataset file.')
    # required parameter
    parser.add_argument('--input_variant_cpg_file', required=True, help='')
    parser.add_argument('--output_variant_cpg_dataset_file', required=True, help='')
    # 解析参数
    args = parser.parse_args()
    # 处理、输出
    variant_cpg_dataset_pldf = process_variant_cpg_file(args.input_variant_cpg_file)
    output_variant_cpg_dataset_pldf(variant_cpg_dataset_pldf, args.output_variant_cpg_dataset_file)


if __name__ == "__main__":
    main_argparse()








































