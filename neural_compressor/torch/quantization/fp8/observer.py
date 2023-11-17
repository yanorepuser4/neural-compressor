import torch
from typing import Tuple
from torch.ao.quantization.observer import UniformQuantizationObserverBase, HistogramObserver, is_per_tensor


# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240*0.9, dtype=torch.float).to('hpu')
E5M2_AMAX = torch.tensor(57344*0.9, dtype=torch.float).to('hpu')


class FP8HistogramObserver(HistogramObserver):
    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        format='E4M3',
    ) -> None:
        # bins: The number of bins used for histogram calculation.
        super(FP8HistogramObserver, self).__init__(
            bins=bins,
            upsample_rate=upsample_rate,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )

    def _get_dst_bin(self, src_bin_begin, src_bin_end, dst_bin_max):
        # get dst bin value
        FP8_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        scale = FP8_amax / dst_bin_max
        if torch.isinf(torch.tensor(scale)):
            scale = torch.tensor(3.4E38)
        tmp = torch.ops.hpu.cast_to_fp8_v2(src_bin_begin, scale, False, False, self.dtype)[0]
        dst_bin_begin = torch.ops.hpu.cast_from_fp8(tmp, None, torch.float32)
        tmp = torch.ops.hpu.cast_to_fp8_v2(src_bin_end, scale, False, False, self.dtype)[0]
        dst_bin_end = torch.ops.hpu.cast_from_fp8(tmp, None, torch.float32)
        # get bin width of dst bin value, dst_bin_begin must contain 0 and the max qvalue.
        dst_bin = list(set(dst_bin_begin.detach().cpu().numpy()))
        dst_bin.sort()
        width_dict = {}
        bin_of_dst_dict = {}
        for i, bin in enumerate(dst_bin):
            bin_of_dst_dict[bin] = i
            if bin == 0:
                width_dict[bin] = {'left': 0, 'right': dst_bin[i+1]}
            elif i == len(dst_bin)-1:
                width_dict[bin] = {'left': dst_bin[i] - dst_bin[i-1],
                                    'right': dst_bin[i] - dst_bin[i-1]}
            else:
                width_dict[bin] = {'left': dst_bin[i] - dst_bin[i-1],
                                    'right': dst_bin[i+1] - dst_bin[i]}
        dst_bin_of_begin = [bin_of_dst_dict[float(i)] for i in dst_bin_begin]
        dst_bin_of_end = [bin_of_dst_dict[float(i)] for i in dst_bin_end]
        left_dst_bin_end_width = [width_dict[float(i)]['left'] for i in dst_bin_end]
        right_dst_bin_begin_width = [width_dict[float(i)]['right'] for i in dst_bin_begin]
        return dst_bin_begin, dst_bin_end, \
                torch.tensor(dst_bin_of_begin), torch.tensor(dst_bin_of_end), \
                torch.tensor(left_dst_bin_end_width), torch.tensor(right_dst_bin_begin_width)

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins
        dst_bin_max = bin_width * (next_end_bin - next_start_bin + 1)

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        src_bin_begin = src_bin * bin_width
        src_bin_end = src_bin_begin + bin_width
        dst_bin_begin, dst_bin_end, \
            dst_bin_of_begin, dst_bin_of_end, \
            left_dst_bin_end_width, right_dst_bin_begin_width = \
                                self._get_dst_bin(src_bin_begin, src_bin_end, dst_bin_max)

        dst_bin_of_begin_center = dst_bin_begin + right_dst_bin_begin_width
        dst_bin_of_end_center = dst_bin_end + left_dst_bin_end_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = right_dst_bin_begin_width

        norm += self._get_norm(delta_begin, delta_end, density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-left_dst_bin_end_width), torch.tensor(right_dst_bin_begin_width), density
        )

        delta_begin = -left_dst_bin_end_width
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(delta_begin, delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha
            next_beta = beta - stepsize

            # find the right bins between the quantile bounds
            # keep the left bins at zero due to fp8 symmetry
            l = 0
            r = end_bin
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) <= (end_bin - r):
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        # use abs due to fp8 symmetry
        x = torch.abs(x)
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float("inf") and max_val == float("-inf")
        if is_uninitialized or same_values:
            min_val, max_val = torch.aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert (
                min_val.numel() == 1 and max_val.numel() == 1
            ), "histogram min/max values must be scalar."
            torch.histc(
                x, self.bins, min=int(min_val), max=int(max_val), out=self.histogram
            )
        else:
            new_min, new_max = torch.aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (
                combined_min,
                combined_max,
                downsample_rate,
                start_idx,
            ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert (
                combined_min.numel() == 1 and combined_max.numel() == 1
            ), "histogram min/max values must be scalar."
            combined_histogram = torch.histc(
                x, self.bins, min=int(combined_min), max=int(combined_max)
            )
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

            self.histogram.detach_().resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.detach_().resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.detach_().resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

class MinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\text{min/max}` is computed as:

    .. math::

        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "MinMaxObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        #print("max :", max_val, "min :", min_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))
