import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from concurrent.futures import ThreadPoolExecutor
import app.config as config

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
#from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())
# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Session store (module-level singleton)
# ---------------------------------------------------------------------------
session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the ChatMessageHistory for *session_id*."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILE_EXAMPLES: Dict[str, list] = {
    "Recruiter": [
        {
            "query": "Can you summarize Maguette's professional experience?",
            "answer": (
                "Maguette MBAYE is a Doctor in AI with strong experience in machine learning, "
                "RAG systems, and full-stack development. She has worked on various AI-driven "
                "projects and is known for combining rigorous research with practical engineering."
            ),
        },
        {
            "query": "What are Maguette's key skills and strengths?",
            "answer": (
                "Maguette excels in Python, NLP, and AI/ML frameworks. She is known for her "
                "analytical mindset, ability to lead complex projects, and her talent for making "
                "technical concepts accessible to non-technical stakeholders."
            ),
        },
        {
            "query": "Can you provide details about Maguette's recent projects?",
            "answer": (
                "Maguette recently built a RAG-based conversational AI chatbot to showcase her "
                "professional profile to recruiters and hiring managers. She also has experience "
                "with vector stores, LLM orchestration, and cloud-based deployments."
            ),
        },
    ],
    "Technical Hiring Manager": [
        {
            "query": "What programming languages and frameworks is Maguette proficient in?",
            "answer": (
                "Maguette is proficient in Python and has hands-on experience with LangChain, "
                "FAISS, OpenAI APIs, FastAPI, and Docker. Her PhD background gives her a deep "
                "understanding of ML model architectures and evaluation."
            ),
        },
        {
            "query": "Can you describe Maguette's experience with AI systems?",
            "answer": (
                "Maguette has built end-to-end RAG pipelines including embedding generation, "
                "vector store indexing, retrieval, reranking with LLM-as-a-judge, and "
                "conversational memory management. She is comfortable with both research and "
                "production-grade implementations."
            ),
        },
        {
            "query": "How does Maguette approach problem-solving in technical projects?",
            "answer": (
                "Maguette combines her research background with pragmatic engineering. She "
                "starts from first principles, benchmarks multiple approaches, and iterates "
                "quickly. She values clean code, clear documentation, and measurable results."
            ),
        },
    ],
    "General": [
        {
            "query": "Can you tell me about Maguette's background?",
            "answer": (
                "Maguette MBAYE is a Doctor in AI, software engineer, and proud mom. She is "
                "passionate about using AI to solve real-world problems and is currently "
                "building innovative AI-powered products."
            ),
        },
        {
            "query": "What are some of Maguette's interests?",
            "answer": (
                "Beyond her technical work, Maguette is passionate about AI ethics, women in "
                "tech, and continuous learning. She enjoys exploring new ideas at the "
                "intersection of research and product."
            ),
        },
        {
            "query": "How can I contact Maguette for professional inquiries?",
            "answer": (
                "You can reach out to Maguette via her LinkedIn profile or email her directly "
                "at maguette@example.com."
            ),
        },
    ],
}

VALID_PROFILES = set(PROFILE_EXAMPLES.keys())

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = """You are MaguetteAgent, an AI assistant specialised in answering \
questions about Maguette MBAYE's CV, projects, and skills.
Maguette is a married Doctor in AI and a mom.

Use the retrieved context below to answer the user's question accurately and concisely.

Rules:
- Answer ONLY using the provided documents.
- Be concise, clear, and honest.
- Always use she/her pronouns when referring to Maguette.
- If information is missing, say so explicitly.
- When relevant, mention the project or CV section.
- You are speaking to recruiters and technical hiring managers — be professional but friendly.

Retrieved context:
{context}

Here are some example Q&A pairs to guide your style:
"""

HUMAN_SUFFIX = "Question: {input}"

# Used by the LLM judge for reranking
RERANKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a relevance judge. Given a question and a document, score the \
document's relevance to the question on a scale of 0 to 10.
Respond with ONLY a single integer between 0 and 10. No explanation, no punctuation.""",
    ),
    ("human", "Question: {question}\n\nDocument: {document}\n\nRelevance score:"),
])


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

class Chatbot:
    """Handles a single conversational turn for MaguetteAgent.

    Parameters
    ----------
    user_message:
        The raw text the user sent.
    profile_option:
        One of ``"Recruiter"``, ``"Technical Hiring Manager"``, or ``"General"``.
    session_id:
        Identifies the conversation thread; defaults to ``"default"``.
    rerank_top_n:
        Number of documents to keep after reranking. Defaults to 3.
    """

    def __init__(
        self,
        user_message: str,
        profile_option: str = "General",
        session_id: str = "default",
        rerank_top_n: int = 3,
    ) -> None:
        if profile_option not in VALID_PROFILES:
            raise ValueError(
                f"Unknown profile_option '{profile_option}'. "
                f"Choose from: {sorted(VALID_PROFILES)}"
            )
        self.user_message = user_message
        self.profile_option = profile_option
        self.session_id = session_id
        self.rerank_top_n = rerank_top_n

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_few_shot_examples_text(self) -> str:
        """Render the few-shot examples for the current profile as plain text."""
        examples = PROFILE_EXAMPLES[self.profile_option]

        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template="Query: {query}\nAnswer: {answer}",
        )

        selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=config.max_length,
        )

        selected = selector.select_examples({"query": self.user_message})
        return "\n\n".join(example_prompt.format(**ex) for ex in selected)

    def _build_chat_prompt(self) -> ChatPromptTemplate:
        """Build a ChatPromptTemplate with system context, history, and human turn."""
        system_message = SYSTEM_PREFIX + self._build_few_shot_examples_text()

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),  # persistent history
                ("human", HUMAN_SUFFIX),
            ]
        )

    # ------------------------------------------------------------------
    # Retrieval & reranking
    # ------------------------------------------------------------------

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Concatenate Document page_content strings."""
        return "\n\n".join(doc.page_content for doc in docs)

    
    def _build_retriever(self):
        """Load the FAISS vectorstore and return a retriever."""

        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(
            config.VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Retrieve more docs than needed so the reranker has room to work
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.MAX_RETRIEVALS},
        )
    
    #Hybrid search with similary and key word matching
    def _build_retriever(self):
        """Hybrid retriever : FAISS (sémantique) + BM25 (mots-clés)."""

        # ── FAISS (semantic) ──
        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(
            config.VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.MAX_RETRIEVALS},
        )

        # ── BM25 (keyword) ──
        # Charge tous les docs du vectorstore pour BM25
        all_docs = list(vectorstore.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config.MAX_RETRIEVALS

        # ── Ensemble 50% FAISS + 50% BM25 ──
        return EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )


    def _rerank_documents(
        self, docs: List[Document], question: str
    ) -> List[Document]:
        """Score each document with an LLM judge and return the top_n best ones.

        Uses gpt-4o-mini at temperature=0 for cheap, deterministic scoring.
        Falls back to score=0.0 if the LLM returns an unexpected response.
        """
        judge_llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",  # lightweight model keeps costs low
        )
        rerank_chain = RERANKING_PROMPT | judge_llm | StrOutputParser()

        scored: List[tuple] = []
        for doc in docs:
            try:
                raw = rerank_chain.invoke({
                    "question": question,
                    "document": doc.page_content,
                })
                score = float(raw.strip())
            except (ValueError, Exception):
                score = 0.0  # safe fallback

            print(f"  [rerank] score={score:.1f} | {doc.page_content[:80]}...")
            scored.append((score, doc))

        # Sort descending and keep top_n
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.rerank_top_n]]
    
    def _rerank_documents(self, docs, question):
        judge_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        rerank_chain = RERANKING_PROMPT | judge_llm | StrOutputParser()

        def score_doc(doc):
            try:
                raw = rerank_chain.invoke({
                    "question": question,
                    "document": doc.page_content
                })
                return float(raw.strip()), doc
            except:
                return 0.0, doc

        # Score tous les docs en parallèle
        with ThreadPoolExecutor(max_workers=len(docs)) as executor:
            scored = list(executor.map(score_doc, docs))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self.rerank_top_n]]
    # ------------------------------------------------------------------
    # Chain assembly & invocation
    # ------------------------------------------------------------------

    def get_answer(self) -> str:
        """Assemble the RAG chain with reranking and return the model's answer."""
        retriever = self._build_retriever()
        prompt = self._build_chat_prompt()
        llm = ChatOpenAI(
            temperature=config.CHAT_MODEL_TEMPERATURE,
            model_name=config.MODEL_NAME,
        )

        # ----------------------------------------------------------------
        # Pipeline:
        #   input ──► retrieve (k docs) ──► rerank (top_n docs)
        #         ──► fill prompt ──► LLM ──► string
        #
        # Keys required by the prompt:
        #   "input"   → human turn
        #   "context" → reranked docs formatted as text
        #   "history" → injected by RunnableWithMessageHistory, passed through here
        # ----------------------------------------------------------------
        rag_chain = (
            {
                "input": RunnableLambda(lambda x: x["input"]),
                "context": RunnableLambda(lambda x: self._format_docs(
                    self._rerank_documents(
                        retriever.invoke(x["input"]),
                        question=x["input"],
                    )
                )),
                # history is injected by RunnableWithMessageHistory before
                # the prompt runs — we just pass it through here
                "history": RunnableLambda(lambda x: x.get("history", [])),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_history.invoke(
            {"input": self.user_message},
            config={"configurable": {"session_id": self.session_id}},
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_response(self) -> str:
        """Build the prompt, run the RAG chain, and return the answer."""
        return self.get_answer()



    def stream_answer(self):
        """Stream the answer token by token."""
        retriever = self._build_retriever()
        prompt    = self._build_chat_prompt()
        llm       = ChatOpenAI(
            temperature=config.CHAT_MODEL_TEMPERATURE,
            model_name=config.MODEL_NAME,
        )

        rag_chain = (
            {
                "input":   RunnableLambda(lambda x: x["input"]),
                "context": RunnableLambda(lambda x: self._format_docs(
                    self._rerank_documents(
                        retriever.invoke(x["input"]),
                        question=x["input"],
                    )
                )),
                "history": RunnableLambda(lambda x: x.get("history", [])),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        yield from chain_with_history.stream(
            {"input": self.user_message},
            config={"configurable": {"session_id": self.session_id}},
        )

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
#if __name__ == "__main__":
#    bot = Chatbot(
 #       user_message="What are Maguette's main technical skills?",
  #      profile_option="Technical Hiring Manager",
  #      session_id="test-session",
  #  )
  #  print(bot.generate_response())
