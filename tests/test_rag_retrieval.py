import unittest
from types import SimpleNamespace

from rag_retrieval import HybridReranker, tokenize


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, doc_id):
        return self.documents[doc_id]


class FakeIndex:
    def __init__(self, dense_results):
        self.dense_results = dense_results
        self.index = SimpleNamespace(ntotal=len(dense_results))
        self.index_to_docstore_id = {i: str(i) for i in range(len(dense_results))}
        self.docstore = FakeDocstore(
            {str(i): doc for i, (doc, _) in enumerate(dense_results)}
        )

    def similarity_search_with_score(self, question, k):
        return self.dense_results[:k]


class HybridRetrievalTests(unittest.TestCase):
    def test_tokenizer_keeps_legal_clause_numbers(self):
        self.assertEqual(tokenize("APP 11.1 and section 26WE"), ["app", "11.1", "and", "section", "26we"])

    def test_bm25_can_promote_a_lexically_precise_dense_candidate(self):
        docs = [
            SimpleNamespace(page_content="general privacy information", metadata={"page": 1}),
            SimpleNamespace(page_content="unrelated compliance material", metadata={"page": 2}),
            SimpleNamespace(page_content="section 15 exact statutory duty", metadata={"page": 3}),
        ]
        dense_results = [(docs[0], 0.2), (docs[1], 0.3), (docs[2], 0.5)]
        reranker = HybridReranker(FakeIndex(dense_results), candidate_k=3)

        dense, hybrid = reranker.retrieve("section 15 statutory duty", k=3)

        self.assertIs(dense[0][0], docs[0])
        self.assertIs(hybrid[0][0], docs[2])


if __name__ == "__main__":
    unittest.main()
