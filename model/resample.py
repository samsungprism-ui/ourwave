import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=False):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialization
        '''
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert(padding in ["reflect", "valid"])

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size - 1) // 2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps -= self.kernel_size - 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:, :, diff_steps // 2:-diff_steps // 2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1)
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        if not self.transpose:
            curr_size = (output_size - 1) * self.stride + 1
        else:
            curr_size = output_size

        if self.padding == "valid":
            curr_size += self.kernel_size - 1

        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

def build_sinc_filter(kernel_size, cutoff):
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M // 2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M // 2)) / (i - M // 2)) * \
                        (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter /= np.sum(filter)
    return filter

