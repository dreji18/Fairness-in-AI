## Dbias

Detecting Bias and ensuring Fairness in AI solutions

[![Downloads](https://static.pepy.tech/personalized-badge/dbias?period=total&units=none&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/dbias)
<a href="https://pypi.org/project/Dbias/">
    <img alt="CI" src="https://img.shields.io/badge/pypi-v0.0.8-orange">
</a>

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
run("Billie Eilish issues apology for mouthing an anti-Asian derogatory term in a resurfaced video.")
```

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

Please find more examples in the notebook section.

## About
This is a collective pipeline comprises of 3 Transformer models to de-bias/reduce amount of bias in news articles. The three models are:
- An English sequence classification model, trained on MBAD Dataset to detect bias and fairness in sentences (news articles). This model was built on top of distilbert-base-uncased model and trained for 30 epochs with a batch size of 16, a learning rate of 5e-5, and a maximum sequence length of 512.
- An Entity Recognition model, which is is trained on MBAD Dataset to recognize the biased word/phrases in a sentence. This model was built on top of roberta-base offered by Spacy transformers.
- A Masked Language model, which is a Pretrained model on English language using a masked language modeling (MLM) objective.

# Author
This model is part of the Research topic "Bias and Fairness in AI" conducted by Deepak John Reji, Shaina Raza. If you use this work (code, model or dataset), 

Please star at:
Bias & Fairness in AI, (2022), GitHub repository, https://github.com/dreji18/Fairness-in-AI

## License
[MIT](https://choosealicense.com/licenses/mit/) License
