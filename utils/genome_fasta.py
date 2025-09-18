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

    # start, end should be in bed format
    def get_sequence_tuple(self, chr_number: str, start: int, end: int, upper_sequence: bool = True):
        # header of fasta file format
        fa_header = '_'.join([chr_number, str(start), str(end)])
        # check start position
        if start < 0:
            fa_header = fa_header + "(start_position_smaller_than_0)"
        # check end position
        chr_fa = self.chr_to_dna_dict[chr_number]
        chr_fa_length = len(chr_fa)
        if end > chr_fa_length:
            fa_header = fa_header + "(end_position_bigger_than_" + str(chr_fa_length) + ")"
        # get sequence
        sequence = chr_fa[start:end]
        if upper_sequence:
            sequence = sequence.upper()
        fa_sequence_tuple = (fa_header, sequence)
        return fa_sequence_tuple

    def get_fa_sequence(self, chr_number: str, start: int, end: int, upper_sequence: bool = True):
        fa_sequence_tuple = self.get_sequence_tuple(chr_number, start, end, upper_sequence)
        # head of fastq file
        fa_header = fa_sequence_tuple[0]
        # sequence
        sequence = fa_sequence_tuple[1]
        fa_sequence = ">" + fa_header + "\n" + sequence + "\n"
        return fa_sequence

    def get_tab_delimited_sequence(self, chr_number: str, start: int, end: int, upper_sequence: bool = True):
        fa_sequence_tuple = self.get_sequence_tuple(chr_number, start, end, upper_sequence)
        # head of fastq file
        fa_header = fa_sequence_tuple[0]
        # sequence
        sequence = fa_sequence_tuple[1]
        fa_sequence = fa_header + "\t" + sequence + "\n"
        return fa_sequence


# test this class
def test_genome_fasta():
    os.chdir('/home/chenfaming/genome/ucsc_hg38')
    file_name = 'hg38.fa'
    genome_fasta_file = GenomeFasta(file_name)
    chr_number = 'chr1'
    start_position = 51631
    end_position = 51631 + 2
    fa_sequence = genome_fasta_file.get_fa_sequence_with_chr_start_end(chr_number, start_position, end_position)
    print(fa_sequence)

def test_genome_fasta_hg19():
    os.chdir('/home/chenfaming/genome/ucsc_hg19')
    file_name = 'hg19.fa'
    genome_fasta_file = GenomeFasta(file_name)
    chr_number = 'chr1'
    start_position = 10643
    end_position = 10643 + 3
    fa_sequence = genome_fasta_file.get_fa_sequence_with_chr_start_end(chr_number, start_position, end_position)
    print(fa_sequence)

if __name__ == '__main__':
    # test_genome_fasta()
    test_genome_fasta_hg19()
