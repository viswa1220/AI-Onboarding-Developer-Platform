import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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
            separators=[
                "\nclass ",
                "\ndef ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
        self._vectorstore = None
        # In-memory client — no SQLite, no tenant issues, no locking
        self._client = chromadb.EphemeralClient()

    # ── Clone & load ─────────────────────────────────────────────────────────
    def clone_and_load(
        self, clone_url: str, branch: str, file_extensions: list[str]
    ) -> list:
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

        try:
            if self._vectorstore is None:
                self._vectorstore = Chroma.from_documents(
                    chunks,
                    embedding=self.embeddings,
                    client=self._client,
                    collection_name="codebase",
                )
            else:
                self._vectorstore.add_documents(chunks)
        except Exception as e:
            # Reset and retry on any DB error
            self._vectorstore = None
            self._client = chromadb.EphemeralClient()
            try:
                self._vectorstore = Chroma.from_documents(
                    chunks,
                    embedding=self.embeddings,
                    client=self._client,
                    collection_name="codebase",
                )
            except Exception as retry_err:
                raise RuntimeError(f"Failed to create vector index: {retry_err}")

    # ── Access ───────────────────────────────────────────────────────────────
    def get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            raise RuntimeError("No vector store found. Index a repo first.")
        return self._vectorstore

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def clear_all(self):
        self._vectorstore = None
        self._client = chromadb.EphemeralClient()
        if os.path.exists(REPOS_DIR):
            shutil.rmtree(REPOS_DIR)