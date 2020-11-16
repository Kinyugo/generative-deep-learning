from operator import itemgetter
from typing import Callable, List, Union, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

_size_2_t = Union[int, Tuple[int, int]]
_list_size_2_t = List[_size_2_t]


class EncoderOutputType(TypedDict):
    """
    Type annotations for the encoder output
    """
    sampled_points: torch.TensorType
    conv_output_shape: Tuple[int, int, int, int]


class Lambda(nn.Module):
    """
    Implements a keras like Lambda layer, 
    that wraps a function into a composable module.
    """
    def __init__(self, function: Callable = lambda x: x):
        super(Lambda, self).__init__()

        self.function = function

    def forward(self, x: torch.TensorType):
        return self.function(x)


class ConvBlock(nn.Module):
    """
    Implements a Sequential stack of layers.

    Consists of:

    - `Conv2d`: 
        functions as a convolutional layer.
    - `LeakyReLU`: 
        functions as an activation layer.
    - `BatchNorm2d`(optional): 
        functions as a regularizing layer.
    - `Dropout`(optional): 
        functions as a regularizing layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t,
                 padding: _size_2_t = 1,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(ConvBlock, self).__init__()

        # the convolutional layer and activation layers are
        # always part of the module list.
        # Other layers; dropout and batch normalization
        # are added dynamically based on the arguments.
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.LeakyReLU()
        ])

        # add an optional batch normalization layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        # add an optional dropout layer
        if use_dropout:
            # Dropout2d randomly zeroes out entire channels as
            # opposed to regular Dropout which zeroes out neurons
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.conv_block(x)


class ConvTransposeBlock(nn.Module):
    """
    Implements a Sequential stack of layers.

    Consists of: 

    - `ConvTranspose2d`: 
        functions as a transposed convolutional layers. 
    - `LeakyReLU`:
        functions as an activation layer.
    - `BatchNorm2d`(optional):
        functions as a regularization layer.
    - `Dropout2d`(optional):
        functions as a regularization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t,
                 padding: _size_2_t = 1,
                 output_padding: _size_2_t = 0,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(ConvTransposeBlock, self).__init__()

        # the transposed convolutional layer and activation layers are
        # always part of the module list.
        # Other layers; dropout and batch normalization
        # are added dynamically based on the arguments.
        layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding),
            nn.LeakyReLU()
        ])

        # optionally add a batch normalization layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        # optionally add a dropout layer
        if use_dropout:
            # Dropout2d randomly zeroes out entire channels
            # as opposed to regular Dropout that zeroes out neurons
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv_transpose_block = nn.Sequential(*layers)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.conv_transpose_block(x)


class Encoder(nn.Module):
    """
    The Encoder takes an image and maps it into a multivariate normal distribution. 
    In one dimension the distribution is defined by the mean and the variance. 

    The Encoder then outputs two values mean and logarithmic variance, which together 
    define a multivariate distribution in the latent space. 

    The Encoder consists of: 
        - `ConvBlock`(s):
            carries out the convolutional operations 
        - `Linear`: 
            outputs the mean of the distribution 
        - `Linear`:
            outputs the logarithmic variance of the distribution
        - `Lambda`(Output Layer):
            takes the mean and logarithmic variance of the distribution and 
            samples points from this using the formula:
                `sample = mean + standard_deviation * epsilon`
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 latent_dim: int = 2,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25):
        super(Encoder, self).__init__()

        # latent_dim(latent dimension) is required for reshape operations
        self.latent_dim = latent_dim

        conv_blocks = nn.ModuleList()
        # append ConvBlock(s) with their configuration
        for i in range(len(kernel_sizes)):
            conv_blocks.append(
                ConvBlock(in_channels=in_channels[i],
                          out_channels=out_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=paddings[i],
                          use_batchnorm=use_batchnorm,
                          use_dropout=use_dropout,
                          dropout_rate=dropout_rate))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # for reshape purposes output layers take in input the same shape as latent dimension
        # and outputs the same shape as the latent dimension
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.log_variance_layer = nn.Linear(latent_dim, latent_dim)

        self.sampling_layer = Lambda(self.sample)

    def sample(self, args: List[torch.TensorType]):
        mean, log_variance = args
        epsilon = torch.randn_like(mean)
        standard_deviation = torch.exp(log_variance / 2)

        return mean + standard_deviation * epsilon

    def forward(self, x: torch.TensorType) -> EncoderOutputType:
        # perform a forward pass through the convolutional blocks
        conv_blocks_output = self.conv_blocks(x)
        # the shape of the convolutional blocks is required later for
        # reshape operations in the encoder
        conv_blocks_output_shape = conv_blocks_output.size()

        # reshape the output from the convolutional blocks to a valid shape
        # for the mean and log_variance layers
        reshaped_conv_blocks_output = conv_blocks_output.view(
            -1, self.latent_dim)

        mean = self.mean_layer(reshaped_conv_blocks_output)
        log_variance = self.log_variance_layer(reshaped_conv_blocks_output)

        # sample from a normal distribution defined by the mean and standard_deviation
        sampled_points = self.sampling_layer([mean, log_variance])

        encoder_output: EncoderOutputType = {
            'sampled_points': sampled_points,
            'conv_output_shape': conv_blocks_output_shape
        }

        return encoder_output


class Decoder(nn.Module):
    """
    The Decoder takes a point the latent space and outputs an image.

    The Decoder consists of:

        - `ConvTransposeBlock`(s):
            carries out the transposed convolutional operation
        - OutputLayer:
            - `ConvTranspose2d`:
                for the transposed convolutiona operation 
            - `Sigmoid`:
                activation for the output layer
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 output_paddings: _list_size_2_t,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(Decoder, self).__init__()

        conv_transpose_blocks = nn.ModuleList()
        # append ConvTransposeBlock(s) with their configuration
        for i in range(len(kernel_sizes) - 1):
            conv_transpose_blocks.append(
                ConvTransposeBlock(in_channels=in_channels[i],
                                   out_channels=out_channels[i],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   use_batchnorm=use_batchnorm,
                                   use_dropout=use_dropout,
                                   dropout_rate=dropout_rate))

        # contains a ConvTranspose2d layer and Sigmoid activation layer.
        output_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels[-1],
                               out_channels=out_channels[-1],
                               kernel_size=kernel_sizes[-1],
                               stride=strides[-1],
                               padding=paddings[-1],
                               output_padding=output_paddings[-1]),
            nn.Sigmoid())

        self.decoder = nn.Sequential(*conv_transpose_blocks, output_block)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.decoder(x)


class VariationalAutoEncoder(nn.Module):
    """
    The VariationalAutoEncoder takes an image passes it through an Encoder that maps the input image
    into a multivariate normal distribution in the latent space and then a Decoder that samples from this
    distribution and outputs an image. 

    The VariationalAutoEncoder consists of: 

    - `Encoder`: 
        Takes an input image and maps it to a latent space.
        Consists of: 
        - `ConvBlock`(s):
            carries out the convolutional operations 
        - `Linear`: 
            outputs the mean of the distribution 
        - `Linear`:
            outputs the logarithmic variance of the distribution
        - `Lambda`(Output Layer):
            takes the mean and logarithmic variance of the distribution and 
            samples points from this using the formula:

            `sample = mean + standard_deviation * epsilon`
    - `Linear`: 
        Takes output from the Encoder and applies the linear function. 

    - `Decoder`:
        Takes a point from the latent space and outputs an image.
        Consists of:
        - `ConvTransposeBlock`(s):
            carries out the transposed convolutional operation
        - OutputLayer:
            - `ConvTranspose2d`:
                for the transposed convolutiona operation 
            - `Sigmoid`:
                activation for the output layer
     
    """
    def __init__(self,
                 enc_in_channels: List[int],
                 enc_out_channels: List[int],
                 enc_kernel_sizes: _list_size_2_t,
                 enc_strides: _list_size_2_t,
                 enc_paddings: _list_size_2_t,
                 dec_in_channels: List[int],
                 dec_out_channels: List[int],
                 dec_kernel_sizes: _list_size_2_t,
                 dec_strides: _list_size_2_t,
                 dec_paddings: _list_size_2_t,
                 dec_output_paddings: _list_size_2_t,
                 latent_dim: int,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        """
        Parameters
        ----------
        - `enc_in_channels: List[int]`:
            Number of input channels for each of the `Conv2d` layers in the encoder.
        - `enc_out_channels: List[int]`:
            Number of output channels for each of the `Conv2d` layers in the encoder.
        - `enc_kernel_sizes: List[Union[int, Tuple[int, int]]]`: 
            Size of the kernels for each of the Conv2d layers in the encoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `enc_strides: List[Union[int, Tuple[int, int]]]`:
            Size of strides for each of the Conv2d layers in the encoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `enc_paddings: List[Union[int, Tuple[int, int]]]`:
            Size of padding for each of the Conv2d layers in the encoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `dec_in_channels: List[int]`:
            Number of input channels for each of the `ConvTranspose2d` layers in the decoder.
        - `dec_out_channels: List[int]`:
            Number of output channels for each of the `ConvTranspose2d` layers in the decoder.
        - `dec_kernel_sizes: List[Union[int, Tuple[int, int]]]`: 
            Size of the kernels for each of the ConvTranspose2d layers in the decoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `dec_strides: List[Union[int, Tuple[int, int]]]`:
            Size of strides for each of the ConvTranspose2d layers in the decoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `dec_paddings: List[Union[int, Tuple[int, int]]]`:
            Size of padding for each of the ConvTranspose2d layers in the decoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `dec_output_paddings: List[Union[int, Tuple[int, int]]]`:
            Size of the output padding for each of the ContTranspose2d layers in the decoder.
            Can be a single integer e.g: `2` or a tuple e.g: `(2, 2)`
        - `latent_dim : int`
            Dimension of the latent space.
        - `use_batch_norm : bool`
            Determines whether BatchNorm2d layers will be included after each of
            the convolutional layers
        - `use_dropout : bool`
            Determines whether Dropout2d layers will be included after each of
            the convolutional layers
        - `dropout_rate : float`
            A probability passed to Dropout2d layers to determine the rate at which
            channels will be dropped.
        """
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encoder(in_channels=enc_in_channels,
                               out_channels=enc_out_channels,
                               kernel_sizes=enc_kernel_sizes,
                               strides=enc_strides,
                               paddings=enc_paddings,
                               latent_dim=latent_dim,
                               use_batchnorm=use_batchnorm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)

        self.decoder_input_layer = nn.Linear(latent_dim, latent_dim)

        self.decoder = Decoder(in_channels=dec_in_channels,
                               out_channels=dec_out_channels,
                               kernel_sizes=dec_kernel_sizes,
                               strides=dec_strides,
                               paddings=dec_paddings,
                               output_paddings=dec_output_paddings,
                               use_batchnorm=use_batchnorm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)

    def forward(self, x) -> torch.TensorType:

        # perform a forward pass through the encoder
        encoder_output: EncoderOutputType = self.encoder(x)
        # extract the contents of the encoder output
        sampled_points, conv_output_shape = itemgetter(
            'sampled_points', 'conv_output_shape')(encoder_output)

        # pass the sampled points through the decoder input
        dec_input = self.decoder_input_layer(sampled_points)
        # reshape the decoder input to the shape of the output of the Conv2d layers in
        # the encoder
        reshaped_dec_input = dec_input.view(conv_output_shape)

        return self.decoder(reshaped_dec_input)


if __name__ == "__main__":

    sample_variational_auto_encoder = VariationalAutoEncoder(
        enc_in_channels=[3, 16, 32],
        enc_out_channels=[16, 32, 64],
        enc_kernel_sizes=[3, 3, 7],
        enc_strides=[1, 2, 1],
        enc_paddings=[1, 1, 1],
        dec_in_channels=[64, 32, 16],
        dec_out_channels=[32, 16, 3],
        dec_kernel_sizes=[7, 3, 3],
        dec_strides=[1, 2, 1],
        dec_paddings=[1, 1, 1],
        dec_output_paddings=[0, 1, 0],
        latent_dim=2,
        use_batchnorm=True,
        use_dropout=True)
    print(sample_variational_auto_encoder)

    sample_variational_auto_encoder_inp = torch.randn((1, 3, 1366, 720))
    sample_variational_auto_encoder_out = sample_variational_auto_encoder(
        sample_variational_auto_encoder_inp)

    print(sample_variational_auto_encoder_out.size())