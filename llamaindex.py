from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# 1) Документи → вузли
docs = SimpleDirectoryReader("./data", recursive=True).load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(docs)

# 2) OpenAI Embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 3) Налаштовуємо ChromaDB як векторне сховище
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("llamaindex_demo")
vector_store = ChromaVectorStore(chroma_collection=collection)

# 4) Створюємо індекс із OpenAI embeddings + ChromaDB
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)

# 5) QueryEngine
query_engine = index.as_query_engine(similarity_top_k=5)
user_query = "What is the role of embeddings in RAG?"
response = query_engine.query(user_query)

print("=== Відповідь LLM ===")
print(response)
