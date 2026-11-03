# Build the lab, one checkpoint at a time

Start from a fresh Python environment. Commands assume the project root and an activated virtual environment; Windows can use the direct Python commands in the main README. No prior cache, generated shop data or presenter knowledge is required.

Each chapter asks you to inspect the evidence produced by your own experiment. Work in retrieval-only mode until the real-provider checkpoint. Model and dataset details change outcomes. Tests use provider doubles only for automated contracts; student-facing answer generation uses actual providers.

- [Run the evidence-only baseline](01-setup.md)
- [Load a document and inspect its records](02-records.md)
- [Implement and inspect character splitting](03-chunks.md)
- [Add BM25 retrieval and inspect rankings](04-keyword.md)
- [Add real embeddings and rank fusion](05-meaning.md)
- [Connect a real answer model and require citations](06-providers.md)
- [Create an OKF contents page and linked topics](07-bundle.md)
- [Enforce review, status and expiry checks](08-review.md)
- [Implement contents-first navigation and evidence admission](09-navigation.md)
- [Run both systems on the same question](10-compare.md)
- [Import an unrelated dataset and reproduce an experiment](11-new-data.md)
- [Extend evaluation, reranking or computation receipts](12-extensions.md)

Before using private data, read SECURITY.md. For supported formats and adapter registration, see ../ARCHITECTURE.md.
