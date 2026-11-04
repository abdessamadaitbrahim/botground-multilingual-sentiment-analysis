# Models

The fine-tuned weights are **not published**: they were trained on confidential data, and each `model.safetensors` file is several hundred megabytes. This folder is ignored by Git except for this file.

## Expected layout

The dashboard and notebook `06_evaluation` load the models from these folders:

```
models/
├── xlm-roberta-base-finetuned/
├── mbert-finetuned/
└── darijaBERT-finetuned/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Regenerate the weights

Each model is a pretrained checkpoint fine-tuned for 3-class sentiment classification (negative, neutral, positive).

| Folder | Base checkpoint | Notebook |
|---|---|---|
| `xlm-roberta-base-finetuned` | `xlm-roberta-base` | `03_finetuning_XLM_RoBERTa.ipynb` |
| `mbert-finetuned` | `bert-base-multilingual-cased` | `04_finetuning_mbert.ipynb` |
| `darijaBERT-finetuned` | `SI2M-Lab/DarijaBERT` | `05_finetuning_DarijaBERT.ipynb` |

The notebooks were run on Google Colab (GPU). They read `train_data.csv` and `test_data.csv` from Google Drive and save the model with `save_pretrained(...)`. Download the saved folders and place them here.

## Load a model

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor("XLM-RoBERTa")
print(predictor.predict_single("merci beaucoup pour votre aide"))
```

## Sharing weights

If weights are ever shared, use a Hugging Face Hub repository or a GitHub Release rather than Git, and only after confirming that sharing a model trained on this data is allowed.
