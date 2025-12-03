import os


class GenomeFasta:
    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
        self.fasta_file_list = []
        self._input_file()
        # dict{chr: dna_sequence}
        self.chr_to_dna_dict = dict()
        self._generate_chr_to_dna_dict()

    def _input_file(self):
        with open(self.fasta_file) as fa_file:
            self.fa_file_list = fa_file.read().split('>')[1:]

    def _generate_chr_to_dna_dict(self):
        for file_line in self.fa_file_list:
            chr_key = file_line.split('\n')[0]
            fa_value = ''.join(file_line.split('\n')[1:])
            self.chr_to_dna_dict[chr_key] = fa_value

    def get_sequence_tuple(self, chr: str, start: int, end: int, upper_sequence: bool = True):
        # header of fasta file format
        fasta_name = '_'.join([chr, str(start), str(end)])
        # check start position
        if start < 0:
            fasta_name = fasta_name + "(start_position_smaller_than_0)"
        # check end position
        chr_fasta = self.chr_to_dna_dict[chr]
        chr_fasta_length = len(chr_fasta)
        if end > chr_fasta_length:
            fasta_name = fasta_name + "(end_position_bigger_than_" + str(chr_fasta_length) + ")"
        # get sequence
        sequence = chr_fasta[start:end]
        if upper_sequence:
            sequence = sequence.upper()
        fasta_sequence_tuple = (fasta_name, sequence)
        return fasta_sequence_tuple

    def get_n_number(self, chr_number, start, end):
        fasta_sequence_tuple = self.get_sequence_tuple(chr_number, start, end, upper_sequence=True)
        # sequence
        sequence = fasta_sequence_tuple[1]
        n_number = sequence.count('N')
        return n_number

