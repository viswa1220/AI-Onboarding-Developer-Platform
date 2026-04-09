import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb

REPOS_DIR = "./repos"


class RepoManager:
    """Handles cloning repos, chunking files, and managing the vector store."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
        )
        # Pure in-memory ChromaDB — no SQLite, no tenant, no locking
        self._chroma_client = chromadb.Client()
        self._collection = self._chroma_client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs = []  # keep raw docs for source display

    # ── Clone & load ─────────────────────────────────────────────────────────
    def clone_and_load(self, clone_url: str, branch: str, file_extensions: list[str]) -> list:
        repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = os.path.join(REPOS_DIR, repo_name)

        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        ext_set = set(file_extensions)
        skip_dirs = {"node_modules", "build", "dist", ".next", "__pycache__", "venv", ".venv", ".git"}

        try:
            loader = GitLoader(
                clone_url=clone_url,
                repo_path=repo_path,
                branch=branch,
                file_filter=lambda fp: (
                    any(fp.endswith(ext) for ext in ext_set)
                    and not any(d in fp for d in skip_dirs)
                    and "package-lock" not in fp
                    and "yarn.lock" not in fp
                ),
            )
            docs = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to clone repo: {e}")

        if not docs:
            raise RuntimeError(
                f"No files found matching {file_extensions} in branch '{branch}'. "
                "Check the branch name and file types."
            )

        for doc in docs:
            source = doc.metadata.get("source", "")
            doc.metadata["repo"] = repo_name
            doc.metadata["file_type"] = Path(source).suffix
            doc.metadata["file_name"] = Path(source).name
        return docs

    # ── Splitting ────────────────────────────────────────────────────────────
    def split_documents(self, docs: list) -> list:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            lines = chunk.page_content.split("\n")
            chunk.metadata["approx_start_line"] = 1
            chunk.metadata["approx_end_line"] = len(lines)
        return chunks

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: list, collection_name: str = "default"):
        texts = [c.page_content for c in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(self._all_docs), len(self._all_docs) + len(chunks))]

        # Sanitize metadata
        metadatas = []
        for c in chunks:
            clean = {}
            for k, v in c.metadata.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            metadatas.append(clean)

        # Get embeddings from OpenAI
        embeddings = self.embeddings.embed_documents(texts)

        # Add to ChromaDB directly
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self._all_docs.extend(chunks)

    # ── Search ───────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> list:
        """Search the collection and return relevant chunks with metadata."""
        query_embedding = self.embeddings.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas"],
        )

        docs = []
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                docs.append({"content": doc_text, "metadata": meta})
        return docs

    # ── Status ───────────────────────────────────────────────────────────────
    def is_indexed(self) -> bool:
        return self._collection.count() > 0

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def clear_all(self):
        self._chroma_client = chromadb.Client()
        self._collection = self._chroma_client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs = []
        if os.path.exists(REPOS_DIR):
            shutil.rmtree(REPOS_DIR)