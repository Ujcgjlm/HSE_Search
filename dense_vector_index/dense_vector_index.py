import faiss
import numpy as np
from sklearn.decomposition import PCA
from bert_embedder import BertEmbedder


class DenseVectorIndex:
    def __init__(self, dim_reduction: str = "pca", n_components: int = 128, embedder=BertEmbedder()):
        self.dim_reduction = dim_reduction
        self.embeddings = []
        self.texts = []
        self.embedder = embedder

        if self.dim_reduction == "pca":
            self.reducer = PCA(n_components=n_components, svd_solver="randomized")

        self.index = None

    def add(self, text):
        self.texts.append(text)

    def build_index(self):
        self.embeddings = self.embedder.get_embeddings(self.texts)

        if self.dim_reduction == "pca":
            self.embeddings = self.reducer.fit_transform(self.embeddings)

        data_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(data_dim)
        self.index.add(np.array(self.embeddings, dtype="float32"))

    def search(self, query, k=5):
        query_embedding = self.embedder.get_embeddings([query])

        if self.dim_reduction == "pca":
            query_embedding = self.reducer.transform(query_embedding)

        _, indices = self.index.search(np.array(query_embedding, dtype="float32"), k)
        results = [self.texts[i] for i in indices[0]]
        return results
