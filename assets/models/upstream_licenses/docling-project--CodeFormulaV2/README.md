---
license: cdla-permissive-2.0
datasets:
- ds4sd/SynthFormulaNet
- ds4sd/SynthCodeNet
tags:
- ocr
- code
- math
- formula
---

# Code Formula Model

The **Code Formula Model** processes an image of a code snippet or formula at 120 DPI and outputs its content.

- **Code Snippets**:  
  The model identifies the programming language and outputs the code repsecting the indendation shown in the given image. The output format will be:<br>
  "<\_\<programming language\>\_> \<content of the image\>"<br>
  Example:<br>
  "<_Java_> System.out.println("Hello World.");"

- **Formulas**:  
  The model generates the corresponding LaTeX code.


This model was trained using the following two datasets:
1. https://huggingface.co/datasets/ds4sd/SynthFormulaNet
2. https://huggingface.co/datasets/ds4sd/SynthCodeNet

# References
```bibtex
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

@article{nassar2025smoldocling,
  title={SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion},
  author={Nassar, Ahmed and Marafioti, Andres and Omenetti, Matteo and Lysak, Maksym and Livathinos, Nikolaos and Auer, Christoph and Morin, Lucas and de Lima, Rafael Teixeira and Kim, Yusik and Gurbuz, A Said and others},
  journal={arXiv preprint arXiv:2503.11576},
  year={2025}
}

```