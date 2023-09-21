<div align="center">

Intel® Neural Compressor Extension for Habana
===========================
<h3> An open-source Python library supporting popular model compression techniques on Habana Gaudi 1&2</h3>

[![python](https://img.shields.io/badge/python-3.7%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-2.2-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/master/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-85%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)

</div>

---
<div align="left">

Intel® Neural Compressor extension for Habana aims to provide popular model compression techniques like FP8 and INT8 on Habana GPUs.

## Getting Started
### Quantization with Python API

```shell
# Install Intel Neural Compressor and TensorFlow
pip install neural-compressor
git clone https://github.com/HabanaAI/DeepSpeed.git

```
```python
from neural_compressor.torch import prepare, convert, quantize, quantize_dynamic
from neural_compressor.config import FP8QuantConfig


```

## Additional Content

* [Contribution Guidelines](./docs/CONTRIBUTING.md)
* [Legal Information](./docs/legal_information.md)
* [Security Policy](./docs/SECURITY.md)

## Research Collaborations

Welcome to raise any interesting research ideas on model compression techniques and feel free to reach us ([inc.maintainers@intel.com](mailto:inc.maintainers@intel.com)). Look forward to our collaborations on Intel Neural Compressor!
