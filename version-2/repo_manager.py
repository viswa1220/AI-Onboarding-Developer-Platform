import os
import shutil
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss

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
        self._index = None       # FAISS index
        self._texts = []         # stored chunk texts
        self._metadatas = []     # stored chunk metadata
        self._dimension = None

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

        # Embed
        emb_list = self.embeddings.embed_documents(texts)
        emb_array = np.array(emb_list, dtype=np.float32)

        # Create or add to FAISS index
        if self._index is None:
            self._dimension = emb_array.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)  # inner product (cosine with normalized vecs)

        # Normalize for cosine similarity
        faiss.normalize_L2(emb_array)
        self._index.add(emb_array)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    # ── Search ───────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> list:
        if self._index is None or self._index.ntotal == 0:
            return []

        query_emb = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_emb)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_emb, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append({
                "content": self._texts[idx],
                "metadata": self._metadatas[idx],
                "score": float(scores[0][i]),
            })
        return results

    # ── Status ───────────────────────────────────────────────────────────────
    def is_indexed(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def clear_all(self):
        self._index = None
        self._texts = []
        self._metadatas = []
        self._dimension = None
        if os.path.exists(REPOS_DIR):
            shutil.rmtree(REPOS_DIR)