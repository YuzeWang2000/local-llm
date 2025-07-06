import requests
import json
import ollama
from ollama import chat
import langchain
import langsmith

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
import os
from core.ollama_client import OllamaAPI
class LangchainOllamaAPI(OllamaAPI):
    BASE_URL = "http://localhost:11434"
    
    def __init__(self, model="gemma3n"):
        self.model = model
        self.llm = OllamaLLM(model=self.model)
        self.generate_context = []  # 生成模式上下文
        self.chat_context = []  # 对话上下文
        self.vector_db = None  # 向量存储
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 嵌入模型
        self.rag_chain = None  # RAG链
        self.chunk_size = 1000  # 文本分割大小
        self.chunk_overlap = 200  # 文本分割重叠大小
        self.persist_directory = "./chroma_db"   # 向量数据库持久化目录
        self.documentes_dir = "./documents" # 文档目录
        self.LANGSMITH_API_KEY = ""  # LangSmith API Key
        self.search_k = 10  # 检索时返回的文档数量
        self.rebuild_index_and_chain()  # 初始化时重建索引和RAG链
        
    # 1. 定义文档加载函数，支持PDF, TXT, DOCX等格式
    def load_documents(self):
        documents = []

        for file in os.listdir(self.documentes_dir):
            file_path = os.path.join(self.documentes_dir, file)

            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())

        return documents
    # 2. 文本分割
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Split by double newlines (paragraphs)
                "\n",    # Split by single newlines
                ". ",    # Split by period followed by space (ensure space to avoid splitting mid-sentence e.g. Mr. Smith)
                "? ",    # Split by question mark followed by space
                "! ",    # Split by exclamation mark followed by space
                "。 ",   # Chinese period followed by space (if applicable)
                "？ ",   # Chinese question mark followed by space (if applicable)
                "！ ",   # Chinese exclamation mark followed by space (if applicable)
                "。\n",  # Chinese period followed by newline
                "？\n",  # Chinese question mark followed by newline
                "！\n",  # Chinese exclamation mark followed by newline
                " ",     # Split by space as a fallback
                ""       # Finally, split by character if no other separator is found
            ],
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)

    # 3. 创建或加载向量数据库
    def get_vector_db(self, chunks):
        """Creates a new vector DB or loads an existing one."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"Loading existing vector database from {self.persist_directory}...")
            try:
                # When loading, ChromaDB will check for dimension compatibility.
                # If EMBEDDING_MODEL_PATH changed leading to a dimension mismatch, this will fail.
                return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            except Exception as e:
                print(f"Error loading existing vector database: {e}.")
                print(f"This might be due to a change in the embedding model and a dimension mismatch.")
                # If loading fails, proceed as if it doesn't exist, but only create if chunks are given later.
                return None # Indicate loading failed or DB doesn't exist in a usable state
        else:
            # Directory doesn't exist or is empty
            if chunks:
                print(f"Creating new vector database in {self.persist_directory}...")
                print(f"Creating Chroma DB with {len(chunks)} chunks...")
                try:
                    vector_db = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    print("Vector database created and persisted.")
                    return vector_db
                except Exception as e:
                    print(f"Error creating new vector database: {e}")
                    raise  # Re-raise the exception if creation fails
            else:
                # No chunks provided and DB doesn't exist/is empty - cannot create.
                print(f"Vector database directory {self.persist_directory} not found or empty, and no chunks provided to create a new one.")
                return None # Indicate DB doesn't exist and cannot be created yet
    
    # 6. 创建RAG检索链（使用新方法）
    def create_rag_chain(self,vector_db):
        client = langsmith.Client(api_key=self.LANGSMITH_API_KEY)
        prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)
        # 创建文档组合链
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        # 创建检索链
        # Using default similarity search. If this fails, and embedding model is good,
        # then advanced retrieval or query transformation might be needed.
        retriever = vector_db.as_retriever(search_kwargs={"k": self.search_k})
        rag_chain = langchain.chains.create_retrieval_chain(retriever, combine_docs_chain)

        return rag_chain
    
    def rebuild_index_and_chain(self):
        """Loads documents, creates/updates vector DB by adding new content, and rebuilds the RAG chain."""

        if self.embeddings is None or self.llm is None:
            return "错误：Embeddings 或 LLM 未初始化。"

        # Ensure documents directory exists
        if not os.path.exists(self.documentes_dir):
            os.makedirs(self.documentes_dir)
            print(f"创建文档目录: {self.documentes_dir}")

        # Step 1: Load documents
        print("加载文档...")
        documents = self.load_documents()
        if not documents:
            print(f"在 {self.documentes_dir} 中未找到文档。")
            # Try to load existing DB even if no new documents are found
            print("尝试加载现有向量数据库...")
            # Pass None for chunks as we are just trying to load
            vector_db = self.get_vector_db(None)
            if vector_db:
                print("没有新文档加载，将使用现有的向量数据库。重新创建 RAG 链...")
                self.rag_chain = self.create_rag_chain(vector_db)
                return "没有找到新文档，已使用现有数据重新加载 RAG 链。"
            else:
                # No documents AND no existing DB
                return "错误：没有文档可加载，且没有现有的向量数据库。"

        # Step 2: Split text
        print("分割文本...")
        chunks = self.split_documents(documents)
        if not chunks:
            print("分割后未生成文本块。")
            # Try loading existing DB if splitting yielded nothing
            print("尝试加载现有向量数据库...")
            vector_db = self.get_vector_db(None)
            if vector_db:
                print("警告：新加载的文档分割后未产生任何文本块。使用现有数据库。")
                self.rag_chain = self.create_rag_chain(vector_db) # Ensure chain is recreated
                return "警告：文档分割后未产生任何文本块。RAG 链已使用现有数据重新加载。"
            else:
                # No chunks AND no existing DB
                return "错误：文档分割后未产生任何文本块，且无现有数据库。"

        # Step 3: Load or Create/Update vector database
        print("加载或更新向量数据库...")
        # Try loading first, even if we have chunks (in case we want to add to it)
        vector_db_loaded = self.get_vector_db(None)

        if vector_db_loaded:
            print(f"向现有向量数据库添加 {len(chunks)} 个块...")
            vector_db = vector_db_loaded # Use the loaded DB
            try:
                # Consider adding only new chunks if implementing duplicate detection later
                vector_db.add_documents(chunks)
                print("块添加成功。")
                # Persisting might be needed depending on Chroma version/setup, often automatic.
                # vector_db.persist() # Uncomment if persistence issues occur
            except Exception as e:
                print(f"添加文档到 Chroma 时出错: {e}")
                # If adding fails, proceed with the DB as it was before adding
                self.rag_chain = self.create_rag_chain(vector_db)
                return f"错误：向向量数据库添加文档时出错: {e}。RAG链可能使用旧数据。"
        else:
            # Database didn't exist or couldn't be loaded, create a new one with the current chunks
            print(f"创建新的向量数据库并添加 {len(chunks)} 个块...")
            try:
                # Call get_vector_db again, this time *with* chunks to trigger creation
                vector_db = self.get_vector_db(chunks)
                if vector_db is None: # Check if creation failed within get_vector_db
                    raise RuntimeError("get_vector_db failed to create a new database.")
                print("新的向量数据库已创建并持久化。")
            except Exception as e:
                print(f"创建新的向量数据库时出错: {e}")
                return f"错误：创建向量数据库失败: {e}"

        if vector_db is None:
            # This should ideally not be reached if error handling above is correct
            return "错误：未能加载或创建向量数据库。"

        # Step 4: Create RAG chain
        print("创建 RAG 链...")
        self.rag_chain = self.create_rag_chain(vector_db)
        print("索引和 RAG 链已成功更新。")
        return "文档处理完成，索引和 RAG 链已更新。"

    # 7. Function to process query using the RAG chain (Modified for Streaming)
    def process_query(self, query):
        """Processes a user query using the RAG chain and streams the answer."""
        if self.rag_chain is None:
            yield "错误：RAG 链未初始化。"
            return

        # --- For Debugging Retrieval ---
        # Uncomment the block below to see what documents are retrieved by the vector DB
        if self.vector_db:
            try:
                retrieved_docs = self.vector_db.similarity_search(query, k=self.search_k)
                print(f"\n--- Retrieved Documents for query: '{query}' ---")
                for i, doc in enumerate(retrieved_docs):
                    # Attempt to get score if retriever supports it (Chroma's similarity_search_with_score)
                    # For basic similarity_search, score might not be directly in metadata.
                    # If using retriever.get_relevant_documents(), score might be present.
                    score = doc.metadata.get('score', 'N/A') # Placeholder, actual score retrieval might differ
                    if hasattr(doc, 'score'): # Check if score attribute exists directly
                        score = doc.score
                    
                    print(f"Doc {i+1} (Score: {score}):")
                    print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
                    print(f"Metadata: {doc.metadata}")
                print("--- End Retrieved Documents ---\n")
            except Exception as e:
                print(f"Error during debug similarity_search: {e}")
        else:
            print("Vector DB not initialized, skipping debug retrieval.")
        # --- End Debugging Retrieval ---

        try:
            print(f"开始处理流式查询: {query}")

            # Directly stream from the RAG chain runnable
            # The input format for create_retrieval_chain is typically {"input": query}
            # The output chunks often contain 'answer' and 'context' keys
            response_stream = self.rag_chain.stream({"input": query})

            full_answer = ""
            # Yield chunks as they arrive. Gradio Textbox updates incrementally.
            print("开始流式生成回答...")
            for chunk in response_stream:
                # Check if the 'answer' key exists in the chunk and append it
                answer_part = chunk.get("answer", "")
                if answer_part:
                    full_answer += answer_part
                    # Debugging output
                    # print(f"Raw answer_part from LLM: '{answer_part}'")
                    # print(f"Yielding to Gradio: '{full_answer}'")
                    yield full_answer # Yield the progressively built answer

            if not full_answer:
                yield "抱歉，未能生成回答。" # Handle cases where stream completes without answer

            print(f"流式处理完成。最终回答: {full_answer}")

        except Exception as e:
            print(f"处理查询时发生错误: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            yield f"处理查询时发生错误: {e}"

    def stream_rag_response(self, prompt):

        for stream_chunk in self.process_query(prompt): # process_query yields full accumulated answer
            # print(f"Stream chunk received: {stream_chunk}")  # Debugging output
            yield stream_chunk
    
    
    def change_model(self, model_name):
        self.model = model_name
        self.llm = OllamaLLM(model=self.model)
        self.reset_context()