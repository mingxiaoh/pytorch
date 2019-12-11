from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
from chainer import testing
# skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.jit
from common_utils import TestCase, run_tests

import warnings


@testing.parameterize(*(testing.product({
    'group': [1, 2, 6],
    'bs': [2],
    'input_channels': [1, 16, 30, 32, 33, 64],
    'output_channels': [1, 16, 17],
    'paddings': [[0, 0], [1, 1]],
    'dilations': [[1, 1], [2, 2]],
    'strides': [[1, 1], [2, 2]],
    'kernels': [[1, 1], [2, 2], [3, 3], [5, 5], [7, 7]],
    'output_sizes': [[1, 1]],
    'bias' : [True, False],
}) + testing.product({
    'group': [2],
    'bs': [1, 4, 16, 64],
    'input_channels': [3],
    'output_channels': [3],
    'paddings': [[1, 1]],
    'dilations': [[1, 1]],
    'strides': [[1, 1]],
    'kernels': [[3, 3]],
    'output_sizes': [[2, 2]],
    'bias' : [True, False],
}) + testing.product({
    'group': [6],
    'bs': [2],
    'input_channels': [3],
    'output_channels': [3],
    'paddings': [[1, 1]],
    'dilations': [[1, 1]],
    'strides': [[1, 1]],
    'kernels': [[3, 3]],
    'output_sizes': [[1, 1], [2, 2], [3, 3], [5, 5], [6, 6], [7, 7], [16, 16], [17, 17], [32, 32], [33, 33], [64, 64], [128, 128]],
    'bias' : [True, False],
})))
# Comment the line below to find out the CI machines having MKL-DNN build disabled
@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnnConv2dExt(TestCase):
    def test_conv2d_ext(self):
        def subtensor(tensor, dim, groups, g):
            if tensor is None:
                return None
            group_size = int(tensor.size(dim) / group)
            return tensor.narrow(dim, group_size * g, group_size).contiguous()

        def thnn_conv(input, weight, k, bias, s, p, d):
            if d[0] > 1 or d[1] > 1:
                return torch._C._nn.slow_conv_dilated2d(input, weight, k, bias, s, p, d)
            else:
                return torch._C._nn.thnn_conv2d(input, weight, k, bias, s, p)

        def thnn_conv_group(input, weight, k, bias, s, p, d, group):
            if group == 1:
                return thnn_conv(input, weight, k, bias, s, p, d)
            else:
                outputs = []
                for g in range(group):
                    input_g = subtensor(input, 1, group, g)
                    weight_g = subtensor(weight, 0, group, g)
                    bias_g = subtensor(bias, 0, group, g)
                    outputs.append(thnn_conv(input_g, weight_g, k, bias_g, s, p, d))
                return torch.cat(outputs, 1)

        def test_single_conv2d(g, bs, isize, ic_g, oc_g, has_bias, p, d, s, k):
            ic, oc = ic_g * g, oc_g * g

            torch.manual_seed(1)
            input = torch.randn((bs, ic, isize[0], isize[1]), dtype=torch.float32, requires_grad=True)
            torch.manual_seed(2)
            weight = torch.randn((oc, ic_g, k[0], k[1]), dtype=torch.float32) * 0.01
            torch.manual_seed(3)
            bias = None if has_bias else torch.randn((oc), dtype=torch.float32)

            # do forward
            self.assertEqual(
                thnn_conv_group(input, weight, k, bias, s, p, d, g),
                torch.mkldnn_convolution(input, weight, bias, p, s, d, g),
                message='input size:{}; group:{}; input channel:{}; output channel:{};'
                        'bias:{}, padding:{}, dilation:{}; stride:{}; kernel:{}'.format(
                        isize, g, ic, oc, has_bias, p, d, s, k)
            )

        group = self.group
        bs = self.bs
        ic = self.input_channels
        oc = self.output_channels
        bias = self.bias
        p = self.paddings
        d = self.dilations
        s = self.strides
        k = self.kernels
        osize = self.output_sizes

        isize = []
        for i in range(len(osize)):
            isize.append(s[i] * (osize[i] - 1) + 1 + d[i] * (k[i] - 1) - 2 * p[i])

        if isize[0] > 0 and isize[1] > 0:
            test_single_conv2d(group, bs, isize, ic, oc, bias, p, d, s, k)
        else:
            warnings.warn(UserWarning("config error, skip..."))

if __name__ == '__main__':
    run_tests()
