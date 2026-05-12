# Class Refactoring Specification

**Goal:** Safely refactor functional code to class-based architecture without breaking existing functionality.

**Principles:**
- Zero breaking changes
- Backward compatibility maintained throughout
- Incremental migration
- Full test coverage at each step
- Rollback capability

---

## 1. Current State Analysis

### 1.1 Existing Functions (to be wrapped)

| Module | Function | Target Class |
|--------|----------|--------------|
| `src/ingestion/embedder.py` | `embed_query()`, `embed_chunks()`, `_get_model()` | `EmbeddingModel` |
| `src/db/chroma_client.py` | `upsert_chunks()`, `load_all_chunks()`, `get_collection()` | `VectorStore` |
| `src/retrieval/bm25_index.py` | `build_bm25_index()`, `bm25_search()` | `BM25Retriever` |
| `src/retrieval/chroma_search.py` | `vector_search()` | `VectorStore` |
| `src/retrieval/cross_encoder.py` | `rerank()`, `_get_model()` | `CrossEncoderReranker` |
| `src/retrieval/pipeline.py` | `retrieval()` | `RetrievalPipeline` |
| `src/generation/chain.py` | `generate()` | `RAGGenerator` |

### 1.2 Global State Issues

```python
# Current anti-patterns (global singletons):
src/ingestion/embedder.py: _model: HuggingFaceEmbeddings | None = None
src/db/chroma_client.py: _vectorstore: Optional[Chroma] = None
src/retrieval/cross_encoder.py: _model = None
```

---

## 2. Target Class Architecture

### 2.1 Class Hierarchy

```
RAGPipeline (orchestrator)
├── VectorStore (data access)
│   └── EmbeddingModel (embeddings)
├── BM25Retriever (keyword search)
├── CrossEncoderReranker (reranking)
└── RAGGenerator (LLM generation)
    └── RetrievalPipeline (retrieval orchestration)
```

### 2.2 Class Specifications

#### **Class: EmbeddingModel**

```python
class EmbeddingModel:
    """Encapsulates embedding generation with lazy loading."""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self._model_name = model_name or Config.EMBEDDING_MODEL
        self._device = device
        self._model: Optional[HuggingFaceEmbeddings] = None
    
    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Lazy-load model on first access."""
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": self._device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._model
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.model.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.embed_documents(texts)
    
    def reset(self):
        """Clear model from memory (for testing)."""
        self._model = None
```

**Backward Compatibility Wrapper:**
```python
# Keep existing functions as thin wrappers
_model_instance: Optional[EmbeddingModel] = None

def _get_model() -> HuggingFaceEmbeddings:
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance.model

def embed_query(text: str) -> list[float]:
    return _get_model().embed_query(text)

def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.embed_documents(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks
```

---

#### **Class: VectorStore**

```python
class VectorStore:
    """Manages ChromaDB operations with proper lifecycle."""
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: EmbeddingModel = None,
    ):
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = persist_directory or str(Config.CHROMA_DIR)
        self.embedding_model = embedding_model or EmbeddingModel()
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load vectorstore on first access."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model.model,
                persist_directory=self.persist_directory,
            )
        return self._vectorstore
    
    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Add or update chunks in the vectorstore."""
        documents = []
        ids = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "page_num": chunk.get("page_num", -1),
                    "chunk_index": chunk.get("chunk_index", -1),
                }
            )
            documents.append(doc)
            ids.append(chunk.get("chunk_id", f"chunk_{len(ids)}"))
        
        self.vectorstore.add_documents(documents=documents, ids=ids)
        return len(chunks)
    
    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Semantic similarity search."""
        results = self.vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        chunks = []
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text": text,
                "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
                **metadata
            })
        return chunks
    
    def load_all(self) -> list[dict]:
        """Load all chunks from the vectorstore."""
        results = self.vectorstore._collection.get()
        chunks = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text": text,
                "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
                **metadata
            })
        return chunks
    
    def reset(self):
        """Clear vectorstore (for testing)."""
        self._vectorstore = None
```

**Backward Compatibility Wrapper:**
```python
_vectorstore_instance: Optional[VectorStore] = None

def _get_vectorstore() -> Chroma:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStore()
    return _vectorstore_instance.vectorstore

def get_collection():
    return _get_vectorstore()._collection

def upsert_chunks(chunks: list[dict]) -> int:
    return _get_vectorstore().upsert_chunks(chunks)

def load_all_chunks() -> list[dict]:
    return _get_vectorstore().load_all()

def reset_client():
    global _vectorstore_instance
    _vectorstore_instance = None
```

---

#### **Class: BM25Retriever**

```python
class BM25Retriever:
    """BM25 keyword search with LangChain integration."""
    
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self._retriever: Optional[BM25RetrieverLC] = None
    
    def build_index(self, chunks: list[dict]):
        """Build BM25 index from chunks."""
        documents = self._chunks_to_documents(chunks)
        self._retriever = BM25RetrieverLC.from_documents(
            documents=documents,
            k=self.top_k
        )
    
    def search(self, query: str, chunks: list[dict]) -> list[dict]:
        """Search the BM25 index."""
        if self._retriever is None:
            self.build_index(chunks)
        
        self._retriever.k = self.top_k
        documents = self._retriever.invoke(query)
        return self._documents_to_chunks(documents)
    
    @staticmethod
    def _chunks_to_documents(chunks: list[dict]) -> list[Document]:
        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "page_num": chunk.get("page_num", -1),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "chunk_id": chunk.get("chunk_id", f"{chunk.get('doc_id')}_chunk_{chunk.get('chunk_index')}"),
                    "source": "bm25",
                }
            )
            for chunk in chunks
        ]
    
    @staticmethod
    def _documents_to_chunks(documents: list[Document]) -> list[dict]:
        return [
            {
                "text": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "page_num": doc.metadata.get("page_num", -1),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "chunk_id": f"{doc.metadata.get('doc_id', 'unknown')}_chunk_{doc.metadata.get('chunk_index', -1)}",
                "source": "bm25",
            }
            for doc in documents
        ]
```

**Backward Compatibility Wrapper:**
```python
def build_bm25_index(chunks: list[dict]) -> BM25RetrieverLC:
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    return retriever._retriever

def bm25_search(bm25: BM25RetrieverLC, query: str, chunks: list[dict], top_k: int) -> list[dict]:
    retriever = BM25Retriever(top_k=top_k)
    retriever._retriever = bm25
    return retriever.search(query, chunks)
```

---

#### **Class: CrossEncoderReranker**

```python
class CrossEncoderReranker:
    """Cross-encoder reranking with lazy loading."""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self._model_name = model_name or Config.RERANKER_MODEL
        self._device = device
        self._model: Optional[CrossEncoder] = None
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy-load model on first access."""
        if self._model is None:
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model
    
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Rerank chunks against a query."""
        if top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        if not chunks:
            return []
        
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        if score_threshold is not None:
            reranked = [(c, s) for c, s in reranked if s >= score_threshold]
        
        return [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in reranked[:top_k]
        ]
    
    def reset(self):
        """Clear model from memory (for testing)."""
        self._model = None
```

**Backward Compatibility Wrapper:**
```python
_reranker_instance: Optional[CrossEncoderReranker] = None

def _get_model() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance.model

def rerank(query: str, chunks: list[dict], top_k: int, score_threshold: float | None = None) -> list[dict]:
    reranker = CrossEncoderReranker()
    return reranker.rerank(query, chunks, top_k, score_threshold)
```

---

#### **Class: RetrievalPipeline**

```python
class RetrievalPipeline:
    """Hybrid retrieval orchestration: BM25 + Vector + RRF + Rerank."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        cross_encoder: CrossEncoderReranker,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder
        self.top_k = top_k
    
    def retrieve(self, query: str, chunks: list[dict]) -> list[dict]:
        """Full retrieval pipeline."""
        # BM25 search
        bm25_results = self.bm25_retriever.search(query, chunks, top_k=20)
        
        # Vector search
        query_embedding = self.vector_store.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=20)
        
        # RRF fusion
        fused = self._rrf_fusion(bm25_results, vector_results, top_k=20)
        
        # Rerank
        reranked = self.cross_encoder.rerank(query, fused, self.top_k)
        
        return reranked
    
    @staticmethod
    def _rrf_fusion(bm25_results: list[dict], vector_results: list[dict], top_k: int) -> list[dict]:
        """Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        all_chunks: dict[str, dict] = {}
        
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)
            all_chunks[chunk_id] = chunk
        
        for rank, chunk in enumerate(vector_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk
        
        ranked_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for chunk_id, score in ranked_chunks[:top_k]:
            chunk = all_chunks[chunk_id].copy()
            chunk["rrf_score"] = score
            fused_results.append(chunk)
        
        return fused_results
```

**Backward Compatibility Wrapper:**
```python
def retrieval(
    query: str,
    chunks: list[dict],
    bm25_index,
    top_k: int = 5,
    lf_retrieval_parent=None,
) -> list[dict]:
    """Legacy wrapper for backward compatibility."""
    pipeline = RetrievalPipeline(
        vector_store=VectorStore(),
        bm25_retriever=BM25Retriever(),
        cross_encoder=CrossEncoderReranker(),
        top_k=top_k,
    )
    return pipeline.retrieve(query, chunks)
```

---

#### **Class: RAGGenerator**

```python
class RAGGenerator:
    """RAG answer generation with citations."""
    
    def __init__(self, llm: ChatGroq, retrieval_pipeline: RetrievalPipeline):
        self.llm = llm
        self.retrieval_pipeline = retrieval_pipeline
    
    def generate(self, query: str, chunks: list[dict]) -> CitedAnswer:
        """Generate answer with citations."""
        # Retrieve top chunks
        top_chunks = self.retrieval_pipeline.retrieve(query, chunks)
        
        # Build citation prompt
        citation_prompt = build_citation_prompt(query, top_chunks)
        
        # Generate answer
        response = self.llm.invoke(citation_prompt)
        answer_text = response.content
        
        # Build sources
        sources = [
            Source(
                doc_id=chunk["doc_id"],
                page_num=chunk["page_num"],
                text=chunk["text"],
            )
            for chunk in top_chunks
        ]
        
        return CitedAnswer(answer=answer_text, sources=sources)
```

**Backward Compatibility Wrapper:**
```python
def generate(query: str, chunks: list[dict], bm25_index) -> CitedAnswer:
    """Legacy wrapper for backward compatibility."""
    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL)
    pipeline = RetrievalPipeline(
        vector_store=VectorStore(),
        bm25_retriever=BM25Retriever(),
        cross_encoder=CrossEncoderReranker(),
    )
    generator = RAGGenerator(llm, pipeline)
    return generator.generate(query, chunks)
```

---

## 3. Migration Strategy

### 3.1 Phase 1: Core Infrastructure (No Breaking Changes)

**Goal:** Create classes while keeping all existing functions as wrappers.

**Steps:**
1. Create `src/core/embedding_model.py` with `EmbeddingModel` class
2. Create `src/core/vector_store.py` with `VectorStore` class
3. Create `src/core/bm25_retriever.py` with `BM25Retriever` class
4. Create `src/core/cross_encoder.py` with `CrossEncoderReranker` class
5. Update existing modules to use classes internally via wrappers
6. Run all existing tests - **must pass 100%**

**Files to Create:**
```
src/core/
├── __init__.py
├── embedding_model.py
├── vector_store.py
├── bm25_retriever.py
└── cross_encoder.py
```

**No Breaking Changes:** All existing function signatures remain identical.

---

### 3.2 Phase 2: Pipeline Classes (No Breaking Changes)

**Goal:** Create pipeline classes with wrappers.

**Steps:**
1. Create `src/core/retrieval_pipeline.py` with `RetrievalPipeline` class
2. Create `src/core/rag_generator.py` with `RAGGenerator` class
3. Update `src/retrieval/pipeline.py` to use class internally
4. Update `src/generation/chain.py` to use class internally
5. Run all existing tests - **must pass 100%**

**Files to Create:**
```
src/core/
├── retrieval_pipeline.py
└── rag_generator.py
```

**No Breaking Changes:** All existing function signatures remain identical.

---

### 3.3 Phase 3: App Integration (No Breaking Changes)

**Goal:** Update `app.py` to use classes directly.

**Steps:**
1. Update `app.py` to instantiate classes at startup
2. Replace function calls with class method calls
3. Keep wrapper functions for any external usage
4. Test Streamlit app end-to-end
5. Run all tests - **must pass 100%**

**No Breaking Changes:** External API unchanged.

---

### 3.4 Phase 4: Deprecation Warnings (No Breaking Changes)

**Goal:** Add deprecation warnings to old functions.

**Steps:**
1. Add `warnings.warn()` to all wrapper functions
2. Document migration guide
3. Update README with new usage examples
4. Run all tests - **must pass 100%**

**Example:**
```python
import warnings

def embed_query(text: str) -> list[float]:
    warnings.warn(
        "embed_query() is deprecated. Use EmbeddingModel.embed_query() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_model().embed_query(text)
```

---

### 3.5 Phase 5: Remove Wrappers (Breaking Changes - Optional)

**Goal:** Remove old functions after deprecation period.

**Steps:**
1. Wait for deprecation period (e.g., 2-3 months)
2. Remove wrapper functions
3. Update any remaining external code
4. Update tests to use classes directly
5. Run all tests - **must pass 100%**

**This phase is optional and can be skipped if backward compatibility is critical.**

---

## 4. Testing Strategy

### 4.1 Test Categories

| Test Type | Purpose | When to Run |
|-----------|---------|-------------|
| **Unit Tests** | Test individual methods | After each class implementation |
| **Integration Tests** | Test class interactions | After Phase 2 |
| **Regression Tests** | Ensure no breaking changes | After every phase |
| **End-to-End Tests** | Test full pipeline | After Phase 3 |

### 4.2 Test Coverage Requirements

- **Minimum 90% code coverage** for new classes
- **100% test pass rate** for existing tests after each phase
- **All edge cases** covered (empty inputs, errors, etc.)

### 4.3 Test Structure

```python
# tests/core/test_embedding_model.py
def test_embedding_model_initialization():
    model = EmbeddingModel()
    assert model._model is None  # Lazy loading

def test_embed_query():
    model = EmbeddingModel()
    embedding = model.embed_query("test")
    assert len(embedding) == 384

def test_backward_compatibility():
    from src.ingestion.embedder import embed_query
    embedding = embed_query("test")
    assert len(embedding) == 384
```

---

## 5. Rollback Plan

### 5.1 Rollback Triggers

- Any test failure
- Performance degradation > 20%
- Unexpected errors in production
- User-reported issues

### 5.2 Rollback Procedure

1. Revert to last known good commit
2. Run all tests to verify
3. Deploy reverted version
4. Investigate failure root cause
5. Fix and retry migration

### 5.3 Rollback Safety

- **Git tags** before each phase
- **Feature branches** for each phase
- **Database backups** (ChromaDB persists to disk)
- **No destructive operations** during migration

---

## 6. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/core/` directory
- [ ] Implement `EmbeddingModel` class
- [ ] Implement `VectorStore` class
- [ ] Implement `BM25Retriever` class
- [ ] Implement `CrossEncoderReranker` class
- [ ] Add backward compatibility wrappers
- [ ] Write unit tests for each class
- [ ] Run all existing tests - **must pass**
- [ ] Git tag: `phase-1-complete`

### Phase 2: Pipeline Classes
- [ ] Implement `RetrievalPipeline` class
- [ ] Implement `RAGGenerator` class
- [ ] Add backward compatibility wrappers
- [ ] Write integration tests
- [ ] Run all existing tests - **must pass**
- [ ] Git tag: `phase-2-complete`

### Phase 3: App Integration
- [ ] Update `app.py` to use classes
- [ ] Test Streamlit app end-to-end
- [ ] Run all existing tests - **must pass**
- [ ] Git tag: `phase-3-complete`

### Phase 4: Deprecation Warnings
- [ ] Add deprecation warnings
- [ ] Update documentation
- [ ] Update README
- [ ] Run all existing tests - **must pass**
- [ ] Git tag: `phase-4-complete`

### Phase 5: Remove Wrappers (Optional)
- [ ] Remove old functions
- [ ] Update tests
- [ ] Run all tests - **must pass**
- [ ] Git tag: `phase-5-complete`

---

## 7. Success Criteria

- ✅ All existing tests pass after each phase
- ✅ No breaking changes to external API
- ✅ Performance unchanged or improved
- ✅ Code coverage > 90% for new code
- ✅ Documentation updated
- ✅ Rollback plan tested

---

## 8. Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 2-3 hours | None |
| Phase 2 | 2-3 hours | Phase 1 |
| Phase 3 | 1-2 hours | Phase 2 |
| Phase 4 | 1 hour | Phase 3 |
| Phase 5 | 1-2 hours | Phase 4 |
| **Total** | **7-11 hours** | - |

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Test failures | Medium | High | Run tests after each change |
| Performance regression | Low | Medium | Benchmark before/after |
| Breaking external code | Low | High | Keep wrappers indefinitely |
| Memory leaks | Low | Medium | Profile memory usage |
| Thread safety issues | Low | High | Use instance-based state |

---

## 10. Next Steps

1. **Review this specification** - Confirm approach
2. **Create feature branch** - `feature/class-refactoring`
3. **Start Phase 1** - Implement core infrastructure
4. **Test thoroughly** - Ensure no regressions
5. **Proceed to Phase 2** - After Phase 1 approval

---

**Ready to proceed?** Start with Phase 1 implementation.
