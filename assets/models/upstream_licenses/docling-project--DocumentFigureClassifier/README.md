---
license: mit
base_model:
- google/efficientnet-b0
---


# EfficientNet-B0 Document Image Classifier

This is an image classification model based on **Google EfficientNet-B0**, fine-tuned to classify input images into one of the following 16 categories:

1. **bar_chart**
2. **bar_code**
3. **chemistry_markush_structure**
4. **chemistry_molecular_structure**
5. **flow_chart**
6. **icon**
7. **line_chart**
8. **logo**
9. **map**
10. **other**
11. **pie_chart**
12. **qr_code**
13. **remote_sensing**
14. **screenshot**
15. **signature**
16. **stamp**

## Citation
If you use this model in your work, please cite the following papers:

```
@article{Tan2019EfficientNetRM,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Mingxing Tan and Quoc V. Le},
  journal={ArXiv},
  year={2019},
  volume={abs/1905.11946}
}

@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {{Docling Technical Report}},
  url={https://arxiv.org/abs/2408.09869},
  eprint={2408.09869},
  doi = "10.48550/arXiv.2408.09869",
  version = {1.0.0},
  year = {2024}
}
```