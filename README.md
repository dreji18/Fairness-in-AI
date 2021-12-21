Detecting Bias and ensuring Fairness in AI solutions


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Dbias.

```bash
pip install Dbias
pip install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl
```

## Usage

```python
from Dbias.text_debiasing import * 

# returns unbiased recommendations for a given sentence fragment.
run("Nevertheless, Trump and other Republicans have tarred the protests as havens for terrorists intent on destroying property.")
```

## About
This is a collective pipeline comprises of 3 Transformer models to de-bias/reduce amount of bias in news articles. The three models are:
- An English sequence classification model, trained on MBAD Dataset to detect bias and fairness in sentences (news articles). This model was built on top of distilbert-base-uncased model and trained for 30 epochs with a batch size of 16, a learning rate of 5e-5, and a maximum sequence length of 512.
- An Entity Recognition model, which is is trained on MBAD Dataset to recognize the biased word/phrases in a sentence. This model was built on top of roberta-base offered by Spacy transformers.
- A Masked Language model, which is a Pretrained model on English language using a masked language modeling (MLM) objective.

# Author
This model is part of the Research topic "Bias and Fairness in AI" conducted by Deepak John Reji, Shaina Raza. If you use this work (code, model or dataset), 

Please cite at:
Bias & Fairness in AI, (2022), GitHub repository, https://github.com/dreji18/Fairness-in-AI/tree/dev

## License
[MIT](https://choosealicense.com/licenses/mit/) License