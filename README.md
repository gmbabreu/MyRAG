# How to Implement This on Your Own

Implementing this graph-based RAG (Retrieval-Augmented Generation) system from scratch involves a mix of traditional NLP techniques, graph theory, and modern language models. Below are the key components, libraries used, and the implementation pipeline.

### Required Libraries

All libraries are found in the requirements file. Here are some of the more important libraries and what they are used for:  
- ```spaCy```: Named entity recognition (NER) and linguistic preprocessing.
- ```SentenceTransformers```: Embedding both text chunks and entities using ```all-MiniLM-L6-v2```.
- ```NetworkX```: Building and visualizing the heterogeneous entity-chunk graph.
- ```scikit-learn```: Cosine similarity calculations.
- ```ollama```: To call LLMs (e.g., ```qwen2.5:3b```) locally using Ollama.
- ```langchain``` + ```pdfplumber```: Loading and chunking text from PDFs.

### Embedding

For embedding, make sure to have the ```all-MiniLM-L6-v2``` from the sentence-transformers library downloaded. It is used to embed both entities and full text chunks for similarity comparisons.

### Chunking Pipeline

In order to generate a graph, you should first chunks the documents. 
In our experiments, the system reads all ```.pdf``` files in a folder and chunks them using Langchain's ```RecursiveCharacterTextSplitter```.
Example PDF Chunking Code:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
import os

def create_chunks(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=2024,
        chunk_overlap=204,
        separators=["\n\n\n", "\n\n", "\n", "."],
        add_start_index=True
    )

    all_text_chunks = []
    for file_path in tqdm(pdf_files):
        raw_docs = PDFPlumberLoader(file_path).load()
        chunks = text_processor.split_documents(raw_docs)
        text_chunks = [chunk.page_content for chunk in chunks]
        all_text_chunks.extend(text_chunks)

    return all_text_chunks
```

### Graph Construction

Once the chunks are created, we can construct a graph:
```python
from MiniGraph import MiniRAGGraph

rag_graph = MiniRAGGraph()
rag_graph.construct_graph(chunks)
```

### Querying & Retrieval

When a user submits a query, we can now retrieve the most relevant chunks:
```python
query = "Who created github"
results = rag_graph.retrieve(query, top_k=3)
for result in results:
    print(result["chunk_id"])
```
```top_k``` refers to the amount of chunks that should be retrieved

### LLM chatting via Ollama

We also include the option to automatically feed the chunks to LLM and chat with it. If you want to chat with the LLM, make sure to have Ollama installed. It is a free and local software for running LLMs. It is called locally using the ```ollama.chat()``` API, and the base model used is ```qwen2.5:3b```.

When using this feature, the user inputs a query and GraphRAG compiles the top-k retrieved chunks as a context prompt. The prompt is then passed to an LLM locally via Ollama to generate a final answer:

```python
rag_graph.llm(query, model="qwen2.5:3b", top_k=5)
```
