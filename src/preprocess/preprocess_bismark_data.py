import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.preprocess import BismarkMethylationFile

def main_argparse():
    parser = argparse.ArgumentParser(description='Preprocess DNA methylation file from Bismark.')
    # required parameter
    parser.add_argument('--input_folder', required=True, help='')
    parser.add_argument('--input_file_suffix', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_log_file', required=True, help='')
    # optional parameter
    parser.add_argument('--reference_cpg_coordinate_file', help='')
    #
    args = parser.parse_args()
    methylation_bed_file = BismarkMethylationFile(
        input_folder=args.input_folder,
        input_file_suffix=args.input_file_suffix,
        reference_cpg_coordinate_file=args.reference_cpg_coordinate_file,
        output_folder=args.output_folder,
        output_log_file=args.output_log_file
    )
    methylation_bed_file.preprocess_all_bed_file()

if __name__ == "__main__":
    main_argparse()