# BERT-Tweet-Sentiment-Classification 🐦


The BERT fine-tuning for the task of Tweets sentiment Classification. Made for educational purposes.

Dataset was taken from <a href='https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv'>Kaggle</a>.

## Data survey

### Labels

Let's take a look first on labels distribution.

![Labels](https://github.com/IOBananaOI/BERT-Tweet-Sentiment-Classification/assets/56229061/7231961a-477c-4b4b-8ea5-b0bbe098cdac)

As we can see, in general there's some kind of balance between Positive and Negative, but other classes are represented much less.

### Text

**Wordcloud before cleaning**

![Wordcloud_before_cleaning](https://github.com/IOBananaOI/BERT-Tweet-Sentiment-Classification/assets/56229061/300e45eb-5db8-41c3-8587-b9a37175c9dc)

As we can see there are a lot of links in the text, so it make sense to remove it before passing into the model.

**Wordcloud after cleaning**

![Wordcloud_after_cleaning](https://github.com/IOBananaOI/BERT-Tweet-Sentiment-Classification/assets/56229061/a058c6a9-02a0-4631-a8b4-756ddc7cc37b)


## Models training

For the task were trained plenty of models. First of all <a href="BertForSequenceClassification">BertForSequenceClassification</a> from Hugging Face 🤗.

The second model was also taken from Transformers lib, but had a little bit another structure.

```rb
class BertTweetClf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.clf = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.Linear(768, 5)
        )
        
    def forward(self, x, attn_mask):
        return self.clf(self.bert(x, attention_mask=attn_mask).pooler_output)
```

For both models was used **bert-base-uncased** version of BERT.

Also both models were trained with and without **gradient clipping** and **text preprocessing** (links and Twitter nicknames starting with '@' deletion) for 20 epochs each.

For optimization in all cases was used <a href="https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html">AdamW</a> optimizer with **learning rate = 0.00002**, **betas = (0.9, 0.999)**.

The threshold for **gradient clipping** was 40.


## Training results

