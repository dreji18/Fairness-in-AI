## Dbias - Detecting Bias and ensuring Fairness in AI solutions

This package is used to detect and mitigate biases in NLP tasks. The model is an end-to-end framework that takes data into a raw form, preprocess it, detect the various types of biases and mitigate them. The output is the text that is free from bias.

[![Downloads](https://static.pepy.tech/personalized-badge/dbias?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/dbias)
<a href="https://pypi.org/project/Dbias/">
    <img alt="CI" src="https://img.shields.io/badge/pypi-v0.1.3-orange">
</a>
<a href="https://youtu.be/Kb-cldoTMeM">
    <img alt="CI" src="https://img.shields.io/badge/Tutorial-Dbias-red">
</a>
<a href="https://youtu.be/Pb_nbveVWQg">
    <img alt="CI" src="https://img.shields.io/badge/Research%20Paper-Video-green">
</a>

For more details, we would suggest reading the paper
- https://link.springer.com/article/10.1007/s41060-022-00359-4 - International Journal of Data Science and Analytics (2022)
- https://arxiv.org/abs/2207.03938 - KDD 2022 Workshop on Data Science and Artificial Intelligence for Responsible Recommendations (DS4RRS)

| Feature  | Output  |
|---|---|
| Text Debiasing  | Returns debiased news recommendations with bias probability |
| Bias Classification | Classifies whether a news article is biased or not with probability |
| Bias Words/Phrases Recognition | Extract Biased words or phrases from the news fragment |
| Bias masking  | Returns the news fragment with biased words masked out |

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Dbias.

```bash
pip install Dbias
pip install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl
```

## Usage

To de-bias a news article
```python
from Dbias.text_debiasing import * 

# returns unbiased recommendations for a given sentence fragment.
run("Billie Eilish issues apology for mouthing an anti-Asian derogatory term in a resurfaced video.", show_plot = True)
```
<img src="plots/bias probability plot.png" alt="drawing" />

To Classify a news article whether it's biased or not
```python
from Dbias.bias_classification import *

# returns classification label for a given sentence fragment.
classifier("Nevertheless, Trump and other Republicans have tarred the protests as havens for terrorists intent on destroying property.")
```

To Recognize the biased words/phrases
```python
from Dbias.bias_recognition import *

# returns extracted biased entities from a given sentence fragment
recognizer("Christians should make clear that the perpetuation of objectionable vaccines and the lack of alternatives is a kind of coercion.")
```

To Mask out the biased portions of a given sentence fragment
```python
from Dbias.bias_masking import *

# returns extracted biased entities from a given sentence fragment
masking("The fact that the abortion rate among American blacks is far higher than the rate for whites is routinely chronicled and mourned.")
```

Please find more examples in the [**notebook section**](https://github.com/dreji18/Fairness-in-AI/tree/main/example%20notebooks).

## About
This is a collective pipeline comprises of 3 Transformer models to de-bias/reduce amount of bias in news articles. The three models are:
- An English sequence classification model, trained on the [**MBIC Dataset**](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE), to detect bias and fairness in sentences (news articles). This model was built on top of distilbert-base-uncased model and trained for 30 epochs with a batch size of 16, a learning rate of 5e-5, and a maximum sequence length of 512.
- An Entity Recognition model, which is is trained on MBIC Dataset to recognize the biased word/phrases in a sentence. This model was built on top of roberta-base offered by Spacy transformers.
- A Masked Language model, which is a Pretrained model on English language using a masked language modeling (MLM) objective.

# Author
This model is part of the Research topic "Bias and Fairness in AI" conducted by Deepak John Reji, Shaina Raza, Chen Ding If you use this work (code, model or data), 

Please cite our [**Research Paper**](https://link.springer.com/article/10.1007/s41060-022-00359-4) 


and please star at: Bias & Fairness in AI, (2022), GitHub repository, https://github.com/dreji18/Fairness-in-AI

## License
[MIT](https://choosealicense.com/licenses/mit/) License
