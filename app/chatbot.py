import os
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

import app.config as config ## app.config   

import httpx
import os

# At the top of chatbot.py, before initialising ChatOpenAI
os.environ["SSL_CERT_FILE"] = "/Users/maguette/Zscaler.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "/Users/maguette/Zscaler.pem"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Session store  (module-level singleton)
# ---------------------------------------------------------------------------
session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the ChatMessageHistory for *session_id*.

    Accepts a plain string instead of a config dict so it is directly
    compatible with RunnableWithMessageHistory's ``get_session_history``
    parameter.
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ---------------------------------------------------------------------------
# Profile definitions  (separated from the class for clarity)
# ---------------------------------------------------------------------------

PROFILE_EXAMPLES: Dict[str, list] = {
    "Recruiter": [
        {
            "query": "Can you summarize Maguette's professional experience?",
            "answer": (
                "Maguette has over 5 years of experience in software development, "
                "specializing in full-stack web development and cloud computing. He has "
                "worked on various projects ranging from e-commerce platforms to scalable "
                "cloud applications."
            ),
        },
        {
            "query": "What are Maguette's key skills and strengths?",
            "answer": (
                "Maguette excels in JavaScript, Python, and cloud technologies like AWS "
                "and Azure. He is known for his problem-solving skills, adaptability, and "
                "ability to work effectively in team environments."
            ),
        },
        {
            "query": "Can you provide details about Maguette's recent projects?",
            "answer": (
                "Recently, Maguette led the development of a scalable e-commerce platform "
                "that integrated multiple payment gateways and implemented advanced security "
                "features. He also contributed to a cloud migration project that improved "
                "system performance by 30%."
            ),
        },
    ],
    "Technical Hiring Manager": [
        {
            "query": "What programming languages and frameworks is Maguette proficient in?",
            "answer": (
                "Maguette is proficient in JavaScript, Python, and Java. He has extensive "
                "experience with frameworks such as React, Node.js, Django, and Spring Boot."
            ),
        },
        {
            "query": "Can you describe Maguette's experience with cloud platforms?",
            "answer": (
                "Maguette has hands-on experience with AWS and Azure, including services "
                "like EC2, S3, Lambda, and Azure Functions. He has successfully led cloud "
                "migration projects and optimised cloud infrastructure for cost and "
                "performance."
            ),
        },
        {
            "query": "How does Maguette approach problem-solving in technical projects?",
            "answer": (
                "Maguette employs a systematic approach to problem-solving, starting with "
                "thorough analysis and understanding of the issue. He collaborates with team "
                "members to brainstorm solutions, implements best practices, and continuously "
                "tests and refines his work to ensure optimal results."
            ),
        },
    ],
    "General": [
        {
            "query": "Can you tell me about Maguette's background?",
            "answer": (
                "Maguette has a diverse background in software development, with experience "
                "in both front-end and back-end technologies. He has worked on various "
                "projects that showcase his versatility and adaptability."
            ),
        },
        {
            "query": "What are some of Maguette's hobbies and interests?",
            "answer": (
                "Outside of work, Maguette enjoys hiking, photography, and exploring new "
                "technologies. He is passionate about continuous learning and often "
                "participates in tech meetups and workshops."
            ),
        },
        {
            "query": "How can I contact Maguette for professional inquiries?",
            "answer": "You can reach out to Maguette via his LinkedIn profile or email him at maguette@example.com.",
        },
    ],
}

VALID_PROFILES = set(PROFILE_EXAMPLES.keys())

SYSTEM_PREFIX = """You are MaguetteAgent, an AI assistant specialised in answering \
questions about Maguette's CV, projects, and skills.

Use the retrieved context below to answer the user's question accurately and concisely.

Rules:
- Answer ONLY using the provided documents.
- Be concise, clear, and honest.
- If information is missing, say so explicitly.
- When relevant, mention the project or CV section.
- You are speaking to recruiters and technical hiring managers — be professional but friendly.

Retrieved context:
{context}

Here are some example Q&A pairs to guide your style:
"""

HUMAN_SUFFIX = "Question: {input}"


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
    """

    def __init__(
        self,
        user_message: str,
        profile_option: str = "General",
        session_id: str = "default",
    ) -> None:
        if profile_option not in VALID_PROFILES:
            raise ValueError(
                f"Unknown profile_option '{profile_option}'. "
                f"Choose from: {sorted(VALID_PROFILES)}"
            )
        self.user_message = user_message
        self.profile_option = profile_option
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_few_shot_examples_text(self) -> str:
        """Render the few-shot examples for the current profile as plain text.

        The rendered text is injected into the system message of the
        ChatPromptTemplate so that message history (via MessagesPlaceholder)
        can coexist with the few-shot examples.
        """
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

        # Render selected examples to a single string
        selected = selector.select_examples({"query": self.user_message})
        rendered = "\n\n".join(example_prompt.format(**ex) for ex in selected)
        return rendered

    def _build_chat_prompt(self) -> ChatPromptTemplate:
        """Build a ChatPromptTemplate that includes:

        1. A system message with the role description, rules, retrieved
           context placeholder (``{context}``), and few-shot examples.
        2. A MessagesPlaceholder for persistent conversation history.
        3. A human turn with the user's input (``{input}``).
        """
        examples_text = self._build_few_shot_examples_text()

        system_message = SYSTEM_PREFIX + examples_text

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),   # ← chat history
                ("human", HUMAN_SUFFIX),
            ]
        )
        return prompt

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_docs(docs) -> str:
        """Concatenate retrieved Document page_content strings."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_retriever(self):
        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(
            config.VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.MAX_RETRIEVALS},
        )

    # ------------------------------------------------------------------
    # Chain assembly & invocation
    # ------------------------------------------------------------------

    def get_answer(self) -> str:
        """Assemble the RAG chain and return the model's answer."""
        retriever = self._build_retriever()
        prompt = self._build_chat_prompt()
        llm = ChatOpenAI(
            temperature=config.CHAT_MODEL_TEMPERATURE,
            model_name=config.MODEL_NAME,
        )

        # ----------------------------------------------------------------
        # Chain:
        #   input dict  ──►  retrieve & format context  ──►  fill prompt
        #              ──►  LLM  ──►  parse to string
        #
        # The input dict passed through the chain must contain:
        #   - "input"   : the user message  (for the human turn)
        #   - "context" : the retrieved docs (for the system message)
        #   - "history" : injected automatically by RunnableWithMessageHistory
        # ----------------------------------------------------------------
        rag_chain = (
            {
                # Pass the user message straight through for the human turn
                "input": RunnablePassthrough() | RunnableLambda(lambda x: x["input"]),
                # Retrieve relevant docs and format them for the system message
                "context": RunnableLambda(
                    lambda x: self._format_docs(retriever.invoke(x["input"]))
                ),
                "history": RunnableLambda(lambda x: x.get("history", [])),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Wrap with message history — history is injected automatically
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,   # plain str → ChatMessageHistory
            input_messages_key="input",                # key in the invoke dict
            history_messages_key="history",            # key in the prompt placeholder
        )

        response: str = chain_with_history.invoke(
            {"input": self.user_message},
            config={"configurable": {"session_id": self.session_id}},
        )
        return response

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_response(self) -> str:
        """Build the prompt, run the RAG chain, and return the answer."""
        return self.get_answer()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
#if __name__ == "__main__":
#    bot = Chatbot(
#        user_message="What are Maguette's main technical skills?",
  #      profile_option="Technical Hiring Manager",
  #      session_id="test-session",
 #   )
  #  print(bot.generate_response())