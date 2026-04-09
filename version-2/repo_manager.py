import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

REPOS_DIR = "./repos"
CHROMA_DIR = "./chroma_db"


class RepoManager:
    """Handles cloning repos, chunking files, and managing the vector store."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=[
                "\nclass ",     # class boundaries
                "\ndef ",       # function boundaries
                "\n\n",         # paragraph breaks
                "\n",           # line breaks
                " ",
                "",
            ],
        )
        self._vectorstore = None

    # ── Clone & load ─────────────────────────────────────────────────────────
    def clone_and_load(
        self, clone_url: str, branch: str, file_extensions: list[str]
    ) -> list:
        repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = os.path.join(REPOS_DIR, repo_name)

        # Remove old clone if exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        ext_set = set(file_extensions)
        skip_dirs = {"node_modules", "build", "dist", ".next", "__pycache__", "venv", ".venv", ".git"}

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

        # Enrich metadata
        for doc in docs:
            source = doc.metadata.get("source", "")
            doc.metadata["repo"] = repo_name
            doc.metadata["file_type"] = Path(source).suffix
            doc.metadata["file_name"] = Path(source).name
        return docs

    # ── Splitting ────────────────────────────────────────────────────────────
    def split_documents(self, docs: list) -> list:
        chunks = self.splitter.split_documents(docs)
        # Add positional metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            content = chunk.page_content
            lines = content.split("\n")
            chunk.metadata["approx_start_line"] = 1
            chunk.metadata["approx_end_line"] = len(lines)
        return chunks

    # ── Sanitize metadata for ChromaDB ───────────────────────────────────────
    def _sanitize_metadata(self, chunks: list) -> list:
        for chunk in chunks:
            clean = {}
            for k, v in chunk.metadata.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            chunk.metadata = clean
        return chunks

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: list, collection_name: str = "default"):
        chunks = self._sanitize_metadata(chunks)
        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DIR,
                collection_name="codebase",
            )
        else:
            self._vectorstore.add_documents(chunks)

    # ── Access ───────────────────────────────────────────────────────────────
    def get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            # Try loading from disk
            if os.path.exists(CHROMA_DIR):
                self._vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=self.embeddings,
                    collection_name="codebase",
                )
            else:
                raise ValueError("No vector store found. Index a repo first.")
        return self._vectorstore

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def clear_all(self):
        self._vectorstore = None
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(REPOS_DIR):
            shutil.rmtree(REPOS_DIR)