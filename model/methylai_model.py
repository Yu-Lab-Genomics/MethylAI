import torch.nn as nn
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from methylai_module import InputBlock, MultiscaleConvBlock, OutputBlock


class MethlyAI(nn.Module):
    def __init__(self, methylai_parameter_dict: dict = None):
        super().__init__()
        # default parameter
        if not methylai_parameter_dict:
            methylai_parameter_dict = {
                'input_block_channel': [20, 190, 30],
                'input_block_kernel_size': [3, 9, 21],
                'input_block_additional_conv_layer': (9, 9),
                'input_block_exponential_activation': True,
                'width': [300, 360, 420, 480, 540, 600],
                'depth': [2, 2, 2, 2, 2, 2],
                'kernel_size': [9, 9, 9, 9, 9, 9],
                'stride': [4, 4, 4, 4, 4, 2],
                'output_block_dims': (1574 * 5, 1574 * 5),
            }
        # check the length of parameter
        assert len(methylai_parameter_dict['input_block_channel']) == 3
        assert len(methylai_parameter_dict['input_block_kernel_size']) == 3
        body_block_length = len(methylai_parameter_dict['width'])
        assert len(methylai_parameter_dict['depth']) == body_block_length
        assert len(methylai_parameter_dict['kernel_size']) == body_block_length
        assert len(methylai_parameter_dict['stride']) == body_block_length
        # input block
        input_block_channel_list = methylai_parameter_dict['input_block_channel']
        input_block_kernel_size_list = methylai_parameter_dict['input_block_kernel_size']
        input_block_additional_conv_layer = methylai_parameter_dict['input_block_additional_conv_layer']
        input_block_exponential_activation = methylai_parameter_dict['input_block_exponential_activation']
        self.input_block = InputBlock(
                input_channel=4,
                output_channel_1=input_block_channel_list[0],
                output_channel_2=input_block_channel_list[1],
                output_channel_3=input_block_channel_list[2],
                conv_kernel_size_1=input_block_kernel_size_list[0],
                conv_kernel_size_2=input_block_kernel_size_list[1],
                conv_kernel_size_3=input_block_kernel_size_list[2],
                additional_conv_layers=input_block_additional_conv_layer,
                exponential_activation=input_block_exponential_activation
        )
        # body parameters
        body_in_channels = sum(input_block_channel_list)
        body_block_list = []
        body_block_args = zip(
            methylai_parameter_dict['width'],
            methylai_parameter_dict['depth'],
            methylai_parameter_dict['kernel_size'],
            methylai_parameter_dict['stride'],
        )
        # for loop construct body
        for (block_width, block_depth, kernel_size, stride) in body_block_args:
            for block_index in range(block_depth):
                body_block_list.append(
                    MultiscaleConvBlock(
                        in_channels=body_in_channels,
                        out_channels=block_width,
                        kernel_size=kernel_size,
                        stride=stride if block_index == 0 else 1,
                    )
                )
                body_in_channels = block_width
        self.body = nn.Sequential(*body_block_list)
        # flatten block
        self.flatten_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        # output block
        output_block_input_dim = sum(methylai_parameter_dict['input_block_channel']) + sum(methylai_parameter_dict['width'])
        output_block_dims = methylai_parameter_dict['output_block_dims']
        self.output_block = OutputBlock(
            input_dim=output_block_input_dim, output_dims_tuple=output_block_dims
        )

    def forward(self, dna, is_return_cpg_embedding: bool = False):
        y1 = self.input_block(dna)
        y2 = self.body(y1)
        cpg_embedding = self.flatten_block(y2)
        dna_methylation_level = self.output_block(cpg_embedding)
        if is_return_cpg_embedding:
            return cpg_embedding, dna_methylation_level
        else:
            return dna_methylation_level










