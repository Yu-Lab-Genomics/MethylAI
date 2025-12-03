import torch
import torch.nn as nn


class ExponentialActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        y = torch.exp(input_tensor)
        return y


class CropLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, output_column_length):
        # 获取输入tensor的形状
        num_batch, num_rows, num_columns = input_tensor.shape
        # 计算起始列的索引
        start_column = (num_columns - output_column_length) // 2
        # 使用索引操作提取中间n列
        middle_columns = input_tensor[:, :, start_column: (start_column + output_column_length)]
        return middle_columns


class InputBlock(nn.Module):
    def __init__(self, input_channel: int = 4,
                 output_channel_1=10, output_channel_2=120, output_channel_3=20,
                 conv_kernel_size_1=3, conv_kernel_size_2=7, conv_kernel_size_3=15,
                 additional_conv_layers=(9, 9), exponential_activation=True):
        super().__init__()
        # parameter
        output_channel_sum = output_channel_1 + output_channel_2 + output_channel_3
        # 1st conv layer
        self.conv1_1 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel_1,
                                 kernel_size=conv_kernel_size_1, padding='same', bias=False)
        self.conv1_2 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel_2,
                                 kernel_size=conv_kernel_size_2, padding='same', bias=False)
        self.conv1_3 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel_3,
                                 kernel_size=conv_kernel_size_3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=output_channel_sum)
        if exponential_activation:
            self.activation = ExponentialActivation()
        else:
            self.activation = nn.GELU()
        # additional conv layer
        conv_block_list = []
        for kernel_size in additional_conv_layers:
            conv_block_list.extend([
                nn.Conv1d(in_channels=output_channel_sum, out_channels=output_channel_sum,
                          kernel_size=kernel_size, padding='same', bias=False),
                nn.BatchNorm1d(output_channel_sum),
                nn.GELU()
            ])
        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, dna):
        y1_1 = self.conv1_1(dna)
        y1_2 = self.conv1_2(dna)
        y1_3 = self.conv1_3(dna)
        y2 = self.activation(self.bn1(torch.cat([y1_1, y1_2, y1_3], dim=-2)))
        y3 = self.conv_block(y2)
        return y3


class MultiscaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True):
        super().__init__()
        # in_channels
        self.in_channels = in_channels
        # basic block
        self.basic_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, bias=False,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=1, bias=False,
                      padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.activation_function = nn.GELU()
        # shortcut conv
        if stride > 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=stride,
                                           stride=stride, bias=bias)
        else:
            self.shortcut_conv = None
        # crop layer
        self.crop_layer = CropLayer()

    def forward(self, x):
        # input
        input_x = x[:, 0:self.in_channels, :]
        # shortcut
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(input_x)
            crop_x = x
        else:
            shortcut = input_x
            crop_x = x[:, self.in_channels:, :]
        # output
        y1 = self.basic_block(input_x)
        y2 = self.activation_function(y1 + shortcut)
        # concat multiscale feature map
        num_batch, num_row, num_col = y2.shape
        crop_x = self.crop_layer(crop_x, num_col)
        y3 = torch.concatenate([y2, crop_x], dim=-2)
        return y3


class OutputBlock(nn.Module):
    def __init__(self, input_dim, output_dims_tuple: tuple):
        super().__init__()
        # 记录input_dim
        self.input_dim = input_dim
        # 设置激活函数
        self.activation_function = nn.ELU
        # 构造output_block
        layer_list = []
        for index, output_dim in enumerate(output_dims_tuple):
            layer_list.append(nn.Linear(input_dim, output_dim))
            # 最后一层之前，使用self.activation_function
            if index < (len(output_dims_tuple) - 1):
                layer_list.append(self.activation_function())
            # 最后一层，使用nn.Sigmoid()
            else:
                layer_list.append(nn.Sigmoid())
            input_dim = output_dim
        self.output_block = nn.Sequential(*layer_list)

    def forward(self, x):
        y = self.output_block(x)
        return y


class MethylAI(nn.Module):
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

