import torch.nn as nn
from  methylai_module import InputBlock, MultiscaleConvBlock, OutputBlock


class MethlyAI(nn.Module):
    def __init__(self, methylai_parameter_dict: dict):
        super().__init__()
        # 默认参数
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
        # 检查参数长度，防止看错
        block_number = len(methylai_parameter_dict['width'])
        assert block_number == len(methylai_parameter_dict['depth'])
        assert block_number == len(methylai_parameter_dict['kernel_size'])
        assert block_number == len(methylai_parameter_dict['stride'])
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
        # body
        body_in_channels = sum(input_block_channel_list)
        body_block_list = []
        body_block_args = zip(
            methylai_parameter_dict['width'],
            methylai_parameter_dict['depth'],
            methylai_parameter_dict['kernel_size'],
            methylai_parameter_dict['stride'],
        )
        # 通过for循环构建body
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
        # for循环结束，构建body
        self.body = nn.Sequential(*body_block_list)
        # flatten block，便于输出embedding
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

    def forward(self, dna):
        y1 = self.input_block(dna)
        y2 = self.body(y1)
        y3 = self.flatten_block(y2)
        y4 = self.output_block(y3)
        return y4















