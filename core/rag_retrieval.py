"""Dense retrieval plus a lightweight BM25 reciprocal-rank-fusion reranker."""

import hashlib
import re

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
RRF_CONSTANT = 60


def tokenize(text):
    return TOKEN_PATTERN.findall((text or "").lower())


def _doc_key(doc):
    metadata = doc.metadata or {}
    identity = "\n".join(
        (doc.page_content, str(metadata.get("source", "")), str(metadata.get("page", "")))
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def _documents_from_faiss(index):
    documents = []
    for position in sorted(index.index_to_docstore_id):
        doc_id = index.index_to_docstore_id[position]
        document = index.docstore.search(doc_id)
        if document is not None:
            documents.append(document)
    return documents


class HybridReranker:
    """Fuse dense and BM25 rankings without another embedding or model call."""

    def __init__(self, index, candidate_k=50):
        self.index = index
        self.candidate_k = candidate_k
        self.documents = _documents_from_faiss(index)
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.documents])

    def retrieve(self, question, k=4):
        dense_results = self.index.similarity_search_with_score(
            question,
            k=self.index.index.ntotal,
        )
        dense_by_key = {_doc_key(doc): (doc, score) for doc, score in dense_results}
        dense_ranks = {_doc_key(doc): rank for rank, (doc, _) in enumerate(dense_results, start=1)}

        lexical_scores = self.bm25.get_scores(tokenize(question))
        lexical_order = sorted(
            (i for i, score in enumerate(lexical_scores) if score > 0),
            key=lambda i: lexical_scores[i],
            reverse=True,
        )
        lexical_ranks = {
            _doc_key(self.documents[i]): rank
            for rank, i in enumerate(lexical_order, start=1)
        }

        candidate_keys = list(dense_ranks)[: self.candidate_k]
        for i in lexical_order[: self.candidate_k]:
            key = _doc_key(self.documents[i])
            if key not in candidate_keys:
                candidate_keys.append(key)

        def rrf_score(key):
            score = 0.0
            if key in dense_ranks:
                score += 1.0 / (RRF_CONSTANT + dense_ranks[key])
            if key in lexical_ranks:
                score += 1.0 / (RRF_CONSTANT + lexical_ranks[key])
            return score

        ranked_keys = sorted(
            candidate_keys,
            key=lambda key: (rrf_score(key), -dense_ranks.get(key, 10**9)),
            reverse=True,
        )
        hybrid_results = [dense_by_key[key] for key in ranked_keys[:k] if key in dense_by_key]
        return dense_results[:k], hybrid_results
