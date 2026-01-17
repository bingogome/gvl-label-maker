# Generative Value Learning Label Maker

Inspired by [OpenGVL](https://github.com/budzianowski/opengvl), based on papers [OpenGVL](https://arxiv.org/abs/2509.17321) and [GVL](https://generative-value-learning.github.io/). This repo focuses on inferences instead of benchmarking.

## Known Issues

```
ImportError("cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'")
```

Add the following two lines to `ENV_SITE_PACKAGES/transformers/utils/import_utils.py`
```
def is_torch_fx_available() -> bool:
    return is_torch_available()
```