# RAG Tutorials

A comprehensive implementation of Retrieval-Augmented Generation (RAG) systems with practical examples, tutorials, and production-ready cod. This project demonstrates how to build end-to-end RAG applications using modern AI frameworks and vector databases.

## 🚀 Features

- **Multi-format Document Loading**: Support for PDF, TXT, CSV, Excel, Word, and JSON files
- **Vector Store Integration**: Built-in support for FAISS and ChromaDB vector databases
- **Advanced Search**: Semantic search with configurable similarity metrics
- **LLM Integration**: Seamless integration with Groq, OpenAI, and other LLM providers
- **Agentic RAG**: Implementation of agent-based RAG systems using LangGraph
- **Typesense Integration**: Hybrid search capabilities with Typesense
- **Interactive Tutorials**: Jupyter notebooks with step-by-step guides 

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Technologies Used](#technologies-used)
- [Features in Detail](#features-in-detail)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hamideshahrabi/rag-tutorials.git
cd rag-tutorials
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## 🏃 Quick Start

### Basic RAG Search

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Load documents
docs = load_all_documents("data")

# Build vector store
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)

# Perform RAG search
rag_search = RAGSearch()
query = "What is attention mechanism?"
summary = rag_search.search_and_summarize(query, top_k=3)
print(summary)
```

### Running the Application

```bash
python app.py
```

## 📁 Project Structure

```
rag-tutorials/
├── src/                    # Core source code
│   ├── data_loader.py      # Document loading utilities
│   ├── embedding.py        # Embedding model management
│   ├── vectorstore.py      # Vector store implementations
│   └── search.py           # RAG search and summarization
├── notebook/               # Tutorial notebooks
│   ├── document.ipynb      # LangChain document components
│   └── pdf_loader.ipynb    # PDF loading examples
├── agenticrag/             # Agentic RAG implementations
│   └── 1-agenticrag.ipynb  # Agent-based RAG tutorial
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 💡 Usage Examples

### Document Loading

```python
from src.data_loader import load_all_documents

# Load all supported documents from a directory
documents = load_all_documents("data")
print(f"Loaded {len(documents)} documents")
```

### Vector Store Operations

```python
from src.vectorstore import FaissVectorStore

# Create and build vector store
store = FaissVectorStore("faiss_store", embedding_model="all-MiniLM-L6-v2")
store.build_from_documents(documents)

# Query similar documents
results = store.query("machine learning", top_k=5)
for result in results:
    print(result["metadata"]["text"])
```

### Custom RAG Pipeline

```python
from src.search import RAGSearch

rag = RAGSearch(
    persist_dir="faiss_store",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gemma2-9b-it"
)

# Search and get summarized results
answer = rag.search_and_summarize(
    "Explain transformer architecture",
    top_k=5
)
print(answer)
```

## 🛠 Technologies Used

- **LangChain**: Framework for building LLM applications
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity search
- **ChromaDB**: Open-source vector database
- **Typesense**: Fast, typo-tolerant search engine
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **Groq**: High-performance LLM inference
- **LangGraph**: Framework for building stateful, multi-actor applications
- **PyPDF/PyMuPDF**: PDF processing libraries

## 📚 Features in Detail

### Document Processing
- Automatic detection and loading of multiple file formats
- Text extraction and preprocessing
- Chunking strategies for large documents
- Metadata preservation

### Vector Search
- Multiple embedding model support
- Configurable similarity metrics
- Efficient indexing and retrieval
- Persistent vector stores

### RAG Pipeline
- Context retrieval from vector stores
- Prompt engineering for summarization
- Multi-LLM provider support
- Configurable top-k retrieval

### Agentic RAG
- Multi-agent systems for complex queries
- State management with LangGraph
- Tool integration capabilities
- Iterative refinement of responses

## 📖 Tutorials

Explore the Jupyter notebooks in the `notebook/` and `agenticrag/` directories for detailed tutorials:

- **Document Components**: Learn about LangChain document structures
- **PDF Loading**: Understand document loading and preprocessing
- **Agentic RAG**: Build agent-based RAG systems
- **Typesense Integration**: Implement hybrid search

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Typesense Documentation](https://typesense.org/docs/)

---

**Note**: Make sure to set up your API keys in the `.env` file before running the application.
