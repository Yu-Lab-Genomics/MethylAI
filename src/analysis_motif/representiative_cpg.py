import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.dataset_tools import HypomethylatedRegion

def main_represent_cpg_argparse():
    parser = argparse.ArgumentParser(description='Generate modeling dataset')
    # required parameter
    parser.add_argument('--complete_dataset_file', required=True, help='')
    parser.add_argument('--col_name', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--model_input_dna_length', type=int, default=18432, help='')


