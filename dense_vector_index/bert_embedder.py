import numpy as np
from transformers import BertModel, BertTokenizer

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy().flatten())
        return np.array(embeddings)
