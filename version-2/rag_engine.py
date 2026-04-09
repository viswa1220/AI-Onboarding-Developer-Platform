from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

SYSTEM_PROMPT = """\
You are an expert developer onboarding assistant. Your job is to help developers
understand, navigate, and work with the codebase that has been indexed.

Rules:
- Answer based ONLY on the provided context. If the context doesn't contain
  enough information, say so clearly.
- When referencing code, mention the file path and approximate location.
- For setup questions, give step-by-step instructions.
- For architecture questions, describe the high-level structure first, then drill down.
- Use markdown formatting: code blocks with language tags, headers, bullet points.
- If the question is ambiguous, ask a clarifying question.

Context from the codebase:
{context}
"""


class RAGEngine:
    """Retrieval-augmented generation engine with chat history support."""

    def __init__(self, repo_manager):
        self.repo_manager = repo_manager

    def ask(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
        top_k: int = 8,
        search_type: str = "mmr",
        temperature: float = 0.2,
    ) -> dict:
        # ── 1. Query rewriting for follow-ups ────────────────────────────────
        effective_query = question
        if chat_history and len(chat_history) > 1:
            effective_query = self._rewrite_query(question, chat_history, temperature)

        # ── 2. Retrieval ─────────────────────────────────────────────────────
        relevant_docs = self.repo_manager.search(effective_query, top_k=top_k)

        # ── 3. Build context ─────────────────────────────────────────────────
        context_parts = []
        sources = []

        for doc in relevant_docs:
            meta = doc["metadata"]
            content = doc["content"]
            file_path = meta.get("source", meta.get("file_name", "unknown"))
            repo = meta.get("repo", "")

            label = f"[{repo}] {file_path}" if repo else file_path
            context_parts.append(f"── {label} ──\n{content}")

            ext = Path(file_path).suffix.lstrip(".")
            lang_map = {
                "py": "python", "js": "javascript", "ts": "typescript",
                "jsx": "jsx", "tsx": "tsx", "json": "json", "md": "markdown",
                "yaml": "yaml", "yml": "yaml", "css": "css",
                "html": "html", "java": "java", "go": "go", "rs": "rust",
            }
            sources.append({
                "file": label,
                "start": meta.get("approx_start_line", "?"),
                "end": meta.get("approx_end_line", "?"),
                "snippet": content[:500],
                "lang": lang_map.get(ext, ""),
            })

        context = "\n\n".join(context_parts)

        # ── 4. Build prompt with history ─────────────────────────────────────
        messages = [("system", SYSTEM_PROMPT)]
        if chat_history:
            for role, content in chat_history[:-1]:
                messages.append((role, content))
        messages.append(("human", "{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)

        # ── 5. Generate ──────────────────────────────────────────────────────
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "sources": sources,
        }

    # ── Query rewriting for multi-turn ───────────────────────────────────────
    def _rewrite_query(
        self, question: str, chat_history: list[tuple[str, str]], temperature: float
    ) -> str:
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the conversation history and a follow-up question, rewrite the "
             "follow-up question to be a standalone question that captures all "
             "necessary context. Return ONLY the rewritten question, nothing else."),
            *[(role, content) for role, content in chat_history[-6:]],
            ("human", "Rewrite this follow-up question as a standalone question: {question}"),
        ])
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        chain = rewrite_prompt | llm
        result = chain.invoke({"question": question})
        return result.content.strip()