from typing import Any, Callable, List, Union, Tuple

import torch
import torch.nn as nn

_size_2_t = Union[int, Tuple[int, int]]
_list_size_2_t = List[_size_2_t]
_tensor_size_3_t = Tuple[torch.TensorType, torch.TensorType, torch.TensorType]


class Lambda(nn.Module):
    """
    Implements a keras like Lambda layer 
    that wraps a function into a module making
    it composable.
    """
    def __init__(self, function: Callable = lambda x: x) -> None:
        super(Lambda, self).__init__()

        self.function = function

    def forward(self, x: torch.TensorType) -> Any:
        return self.function(x)


class Flatten(nn.Module):
    """
    Implements a keras like Flatten layer,
    where only the batch dimension is maintained 
    and all the other dimensions are flattened. 
    i.e reshapes a tensor from (N, *,...) -> (N, product_of_other_dimensions)
    """
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # maintain the batch dimension
        # and concatenate all other dimensions
        return torch.flatten(x, start_dim=1)


class UnFlatten(nn.Module):
    """
    Implements a reshape operation that transforms a 2d
    tensor into a 4d tensor. 
    i.e reshapes a tensor from (N, channels * width * height) -> (N, channels, width, height)
    """
    def __init__(self, num_channels: int) -> None:
        super(UnFlatten, self).__init__()

        self.num_channels = num_channels

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        batch_size = x.size(0)
        width_height_dim = int((x.size(1) // self.num_channels)**0.5)

        return torch.reshape(x, (batch_size, self.num_channels,
                                 width_height_dim, width_height_dim))


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
    The Encoder takes a data point and applies convolutional 
    operations to it.
    Implements a sequential stack of ConvBlock(s).

    Consists of:

    - `ConvBlock`(s):
        carries out the convolutional operations
    - `Flatten`:
        reshapes the 4d output tensor of the convolutional layers 
        into 2d tensor
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(Encoder, self).__init__()

        # initialize conv blocks
        conv_blocks = nn.ModuleList()
        # append ConvBlock(s) whith their configuration
        for i in range(len(kernel_sizes)):
            conv_blocks.append(
                ConvBlock(in_channels=in_channels[i],
                          out_channels=out_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=paddings[i],
                          use_batchnorm=use_batch_norm,
                          use_dropout=use_dropout,
                          dropout_rate=dropout_rate))

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.flatten = Flatten()

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # perform a forward pass through the convolutional blocks
        # (N, C, W, H) -> (N, C, W, H)
        conv_blocks_output = self.conv_blocks(x)

        # flatten the output from the conv_blocks
        # (N, C, W, H) -> (N, C * W * H)
        return self.flatten(conv_blocks_output)


class BottleNeck(nn.Module):
    """
    Implements the layers that output parameters defining the 
    latent space as well as the sampling points from the latent space.

    The BottleNeck consists of:

    - `Linear`(Mean Layer):
        takes in features extracted by the encoder and outputs the mean of the
        latent space distribution.
    - `Linear`(Log Variance Layer):
        takes in features extracted by the encoder and outputs the logarithmic variance
        of the latent space distribution
    - `Lambda`(Reparametarization/Sampling Layer):
        takes the mean and logarithmic variance of the distribution and 
        samples points from this using the formula:
        `sample = mean + standard_deviation * epsilon`
        where epsilon is drawn from a standard normal distribution i.e `epsilon ~ N(0, I)`
    """
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super(BottleNeck, self).__init__()

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_variance = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Lambda(self.sample)

    def forward(self, x: torch.TensorType) -> _tensor_size_3_t:
        # outputs the mean of the distribution
        # mu: (N, latent_dim)
        mu = self.fc_mu(x)
        # outputs the log_variace of the distribution
        # log_variance: (N, latent_dim)
        log_variance = self.fc_log_variance(x)

        # sample z from a distribution
        z = self.sampling([mu, log_variance])

        return z, mu, log_variance

    def sample(self, args):
        mu, log_variance = args

        std = torch.exp(log_variance / 2)
        # define a distribution q with the parameters mu and std
        q = torch.distributions.Normal(mu, std)
        # sample z from q
        z = q.rsample()

        return z


class Decoder(nn.Module):
    """
    The Decoder takes a sampled point from the latent space and reconstructs 
    a data point from the sampled point. 

    The Decoder consists of: 

    - `Linear`(Decoder Input): 
        takes the sampled point and applies a linear transformation to it. 
        This is necessary for reshape operation that is carried out after this layer
        to make sure that the input shape to the ConvTranspose2d layer of the Decoder matches
        the shape of the last Conv2d layer in the Encoder.
    - `UnFlatten`:
        takes a 2d tensor and reshapes it into a 4d tensor. 
        In this case the sampled points are reshaped into a shape that
        can be passed into the ConvTransposeBlock(s).
    - `ConvTransposeBlock`(s):
        carries out the transposed convolutional operation as well as 
        regularization by using batch normalization and dropout layers.
    - OutputBlock: 

        - `ConvTranspose2d`:
            carries out the transposed convolutional operation.
        - `Sigmoid`:
            activation for the output layer. 
            Maps the tensors to the range [0, 1].
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 output_paddings: _list_size_2_t,
                 latent_dim: int,
                 hidden_dim: int,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(Decoder, self).__init__()

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # number of channels should match the number of
        # channels of the first ConvTranspose2d layer
        self.unflatten = UnFlatten(in_channels[0])

        # initialize conv_transpose blocks
        conv_transpose_blocks = nn.ModuleList()
        # append ConvTransposeBlock(s) with their configuration
        # the last configuration is used for the output layer and thus
        # excluded from the list.
        for i in range(len(kernel_sizes) - 1):
            conv_transpose_blocks.append(
                ConvTransposeBlock(in_channels=in_channels[i],
                                   out_channels=out_channels[i],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   use_batchnorm=use_batch_norm,
                                   use_dropout=use_dropout,
                                   dropout_rate=dropout_rate))
        self.conv_transpose_blocks = nn.Sequential(*conv_transpose_blocks)

        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels[-1],
                               out_channels=out_channels[-1],
                               kernel_size=kernel_sizes[-1],
                               stride=strides[-1],
                               padding=paddings[-1],
                               output_padding=output_paddings[-1]),
            nn.Sigmoid())

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # perform a linear transformation on the input
        # (N, latent_dim) -> (N, hidden_dim)
        x = self.decoder_input(x)

        # reshape the decoders input
        # (N, hidden_dim) -> (N, C, W, H)
        x = self.unflatten(x)

        # perform a forward pass through the conv_transpose_blocks
        # (N, C_in, W_in, H_in) -> (N, C_out, W_out, H_out)
        conv_transpose_output = self.conv_transpose_blocks(x)

        # perform a forward pass through the output block
        # (N, C_in, W_in, H_in) -> (N, C_out, W_out, H_out)
        return self.output_block(conv_transpose_output)


class VariationalAutoEncoder(nn.Module):
    """
    The VariationalAutoEncoder takes a data point, passes it through an Encoder,
    and outputs mean and log variance of a multivariate gaussian distribution in the latent space. 
    The Encoder models the conditional distribution p(z|x).

    Using the mean and log variance the reparameterization trick is then applied 
    to sample a point from the normal distribution defined by the mean and variance. 

    The sampled point is then passed through the Decoder that then attempts to reconstruct
    a data point from the sampled point. 
    The Decoder models the conditional distribution p(x|z).

    The VariationalAutoEncoder consists of:

    - `Encoder`:
        Extracts meaningful features from the data.
        Consists of:

        - `ConvBlock`(s):
        carries out the convolutional operations
        - `Flatten`:
            reshapes the 4d output tensor of the convolutional layers 
            into 2d tensor
    - `BottleNeck`:
        Models the latent space of the VariationalAutoEncoder, and 
        samples points from this latent space.

        - `Linear`(Mean Layer):
            takes in features extracted by the encoder and outputs the mean of the
            latent space distribution.
        - `Linear`(Log Variance Layer):
            takes in features extracted by the encoder and outputs the logarithmic variance
            of the latent space distribution
        - `Lambda`(Reparametarization/Sampling Layer):
            takes the mean and logarithmic variance of the distribution and 
            samples points from this using the formula:
            `sample = mean + standard_deviation * epsilon`
            where epsilon is drawn from a standard normal distribution i.e `epsilon ~ N(0, I)`
    - `Decoder`: 
        takes a point sampled from the latent space and reconstructs a data point.
        Consists of:

        - `Linear`(Decoder Input): 
            takes the sampled point and applies a linear transformation to it. 
            This is necessary for reshape operation that is carried out after this layer
            to make sure that the input shape to the ConvTranspose2d layer of the Decoder matches
            the shape of the last Conv2d layer in the Encoder.
        - `UnFlatten`:
            takes a 2d tensor and reshapes it into a 4d tensor. 
            In this case the sampled points are reshaped into a shape that
            can be passed into the ConvTransposeBlock(s).
        - `ConvTransposeBlock`(s):
            carries out the transposed convolutional operation as well as 
            regularization by using batch normalization and dropout layers.
        - OutputBlock: 

            - `ConvTranspose2d`:
                carries out the transposed convolutional operation.
            - `Sigmoid`:
                activation for the output layer. 
                Maps the tensors to the range [0, 1].
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
                 latent_dim: int = 2,
                 data_dim: int = 512,
                 use_batch_norm: bool = False,
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
        - `data_dim : int`
            Dimension of the input data point. i.e for an image this would be
            the height and width of the image.
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

        # instantiate the encoder part.
        self.encoder = Encoder(in_channels=enc_in_channels,
                               out_channels=enc_out_channels,
                               kernel_sizes=enc_kernel_sizes,
                               strides=enc_strides,
                               paddings=enc_paddings,
                               use_batch_norm=use_batch_norm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)

        # to calculate the hidden dim the size of the last dimension
        # of the encoder must be known, hence a sample forward pass is
        # done on the encoder to determine this.
        sample_input = torch.randn((1, enc_in_channels[0], data_dim, data_dim))
        hidden_dim = self.encoder(sample_input).size(-1)

        # initialize the bottleneck layer
        self.bottle_neck = BottleNeck(latent_dim, hidden_dim)

        # initalize the decoder
        self.decoder = Decoder(in_channels=dec_in_channels,
                               out_channels=dec_out_channels,
                               kernel_sizes=dec_kernel_sizes,
                               strides=dec_strides,
                               paddings=dec_paddings,
                               output_paddings=dec_output_paddings,
                               latent_dim=latent_dim,
                               hidden_dim=hidden_dim,
                               use_batch_norm=use_batch_norm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)

    def forward(self, x: torch.TensorType) -> _tensor_size_3_t:
        # extract meaningful features from the data
        # by  performing a forward pass through the encoder
        encoder_output = self.encoder(x)

        # using features extracted from the data obtain mean and log_variance
        # that will be used to define the distribution of the latent space.
        # perform a forward pass through the bottle neck
        z, mu, log_variance = self.bottle_neck(encoder_output)

        # reconstruct the data points from by using the sampled points
        # by performing a forward pass through the decoder.
        x_hat = self.decoder(z)

        return z, mu, log_variance, x_hat


if __name__ == "__main__":
    vae = VariationalAutoEncoder(enc_in_channels=[3, 32, 64],
                                 enc_out_channels=[32, 64, 128],
                                 enc_kernel_sizes=[3, 3, 3],
                                 enc_strides=[1, 2, 1],
                                 enc_paddings=[1, 1, 1],
                                 dec_in_channels=[128, 64, 32],
                                 dec_out_channels=[64, 32, 3],
                                 dec_kernel_sizes=[3, 3, 3],
                                 dec_strides=[1, 2, 1],
                                 dec_paddings=[1, 1, 1],
                                 dec_output_paddings=[0, 1, 0],
                                 latent_dim=16,
                                 data_dim=32,
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=0.20)
    print(vae)

    vae_input = torch.randn((1, 3, 32, 32))
    z, mu, log_variance, x_hat = vae(vae_input)

    def count_model_parameters(model: nn.Module):
        return sum(param.numel() for param in model.parameters()
                   if param.requires_grad)

    print(f'Z shape: {z.size()}')
    print(f'Mean Shape: {mu.size()}')
    print(f'Log variance shape: {log_variance.size()}')
    print(f'X_hat shape: {x_hat.size()}')
    print(f'Number of model parameters: {count_model_parameters(vae)}')
