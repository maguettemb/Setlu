
"""
evaluate.py
-----------
Script d'évaluation minimal du pipeline RAG de MaguetteAgent.

Métriques calculées pour chaque question :
  - Latence        : temps de réponse en secondes
  - Faithfulness   : la réponse est-elle fidèle aux documents récupérés ? (0-10)
  - Relevancy      : la réponse répond-elle à la question ? (0-10)
  - Completeness   : la réponse est-elle complète ? (0-10)
  - Score global   : moyenne des 3 métriques LLM

Un rapport JSON et un résumé console sont générés à la fin.

Usage:
    python -m app.evaluate
    # ou
    python app/evaluate.py
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.chatbot import Chatbot
import app.config as config
# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
import os 
import ssl
import httpx

load_dotenv()

# Chemin absolu vers le cert Zscaler
_CERT = "/Users/maguette/zscaler.pem"

# Créer un contexte SSL custom avec le cert Zscaler
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.load_verify_locations(_CERT)

# Forcer httpx à utiliser ce contexte
_http_client  = httpx.Client(verify=_CERT)
_async_client = httpx.AsyncClient(verify=_CERT)

# Patcher les variables d'environnement pour openai/httpx
os.environ["SSL_CERT_FILE"]      = _CERT
os.environ["REQUESTS_CA_BUNDLE"] = _CERT
# ── FIN DU PATCH ──

# S'assurer que le cert est bien chargé avant httpx
#cert = os.getenv("SSL_CERT_FILE", "")
#if cert:
#    os.environ["SSL_CERT_FILE"] = cert
#    os.environ["REQUESTS_CA_BUNDLE"] = cert

# ---------------------------------------------------------------------------
# Jeu de test
# Format : { "question": str, "expected_keywords": List[str], "profile": str }
# expected_keywords = mots-clés qu'on s'attend à trouver dans la réponse
# ---------------------------------------------------------------------------
TEST_SET = [
    {
        "question": "What are Maguette's main technical skills?",
        "expected_keywords": ["python", "langchain", "rag", "nlp", "ai", "machine learning"],
        "profile": "Technical Hiring Manager",
    },
    {
        "question": "Can you summarize Maguette's professional background?",
        "expected_keywords": ["doctor", "phd", "ai", "experience", "research"],
        "profile": "Recruiter",
    },
    {
        "question": "What projects has Maguette worked on?",
        "expected_keywords": ["chatbot", "rag", "vectorstore", "faiss"],
        "profile": "Technical Hiring Manager",
    },
    {
        "question": "How can I contact Maguette?",
        "expected_keywords": ["linkedin", "email", "maguette"],
        "profile": "General",
    },
    {
        "question": "What is Maguette's educational background?",
        "expected_keywords": ["doctor", "phd", "ai", "university"],
        "profile": "Recruiter",
    },
]

# ---------------------------------------------------------------------------
# LLM Judge prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an evaluation judge. Score the FAITHFULNESS of an answer on a scale of 0 to 10.
Faithfulness = the answer is grounded in the provided context and does not hallucinate.
- 10 : every claim in the answer is supported by the context
- 5  : some claims are supported, some are not
- 0  : the answer contradicts or ignores the context entirely
Respond with ONLY a single integer between 0 and 10.""",
    ),
    (
        "human",
        "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nFaithfulness score:",
    ),
])

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an evaluation judge. Score the RELEVANCY of an answer on a scale of 0 to 10.
Relevancy = the answer directly addresses the question asked.
- 10 : the answer fully and directly answers the question
- 5  : the answer is partially relevant
- 0  : the answer is completely off-topic
Respond with ONLY a single integer between 0 and 10.""",
    ),
    (
        "human",
        "Question: {question}\n\nAnswer: {answer}\n\nRelevancy score:",
    ),
])

COMPLETENESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an evaluation judge. Score the COMPLETENESS of an answer on a scale of 0 to 10.
Completeness = the answer covers all important aspects of the question.
- 10 : the answer is thorough and covers all key points
- 5  : the answer covers some aspects but misses important details
- 0  : the answer is too vague or incomplete to be useful
Respond with ONLY a single integer between 0 and 10.""",
    ),
    (
        "human",
        "Question: {question}\n\nAnswer: {answer}\n\nCompleteness score:",
    ),
])


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

def calculate_latency(chatbot: Chatbot) -> tuple[str, float]:
    """Lance le chatbot et mesure le temps de réponse en secondes.
    
    Retourne (réponse, latence_en_secondes).
    """
    start = time.perf_counter()         # horloge haute précision
    response = chatbot.generate_response()
    latency = time.perf_counter() - start
    return response, round(latency, 3)


def calculate_precision(response: str, expected_keywords: List[str]) -> float:
    """Calcule la précision par présence de mots-clés attendus dans la réponse.

    Précision = nombre de mots-clés trouvés / nombre total de mots-clés attendus.
    Retourne un score entre 0.0 et 1.0.
    """
    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return round(found / len(expected_keywords), 2) if expected_keywords else 0.0


def llm_score(prompt: ChatPromptTemplate, variables: Dict[str, str]) -> float:
    """Envoie un prompt au LLM judge et retourne le score en float.
    
    Retourne 0.0 en cas de réponse inattendue.
    """
    judge = ChatOpenAI(temperature=0, model_name=config.JUDGE_MODEL)
    chain = prompt | judge | StrOutputParser()
    try:
        raw = chain.invoke(variables)
        return float(raw.strip())
    except (ValueError, Exception):
        return 0.0


def evaluate_single(test_case: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Évalue un seul test case et retourne toutes les métriques.

    Étapes :
    1. Lance le chatbot et mesure la latence
    2. Calcule la précision par mots-clés
    3. Demande au LLM judge de scorer faithfulness, relevancy, completeness
    4. Calcule le score global (moyenne des 3 métriques LLM)
    """
    question = test_case["question"]
    profile = test_case["profile"]
    expected_keywords = test_case["expected_keywords"]

    print(f"\n{'─' * 55}")
    print(f"  Question : {question}")
    print(f"  Profile  : {profile}")
    print(f"{'─' * 55}")

    # 1. Latence + réponse
    chatbot = Chatbot(
        user_message=question,
        profile_option=profile,
        session_id=session_id,
    )
    response, latency = calculate_latency(chatbot)
    print(f"  Réponse  : {response[:120]}{'...' if len(response) > 120 else ''}")
    print(f"  Latence  : {latency}s")

    # 2. Précision par mots-clés
    precision = calculate_precision(response, expected_keywords)
    print(f"  Précision (keywords) : {precision:.0%} "
          f"({int(precision * len(expected_keywords))}/{len(expected_keywords)} mots-clés trouvés)")

    # 3. Scores LLM judge
    # Pour faithfulness on a besoin du contexte — on utilise la réponse comme proxy
    # (dans une version avancée, on logguerait le contexte réel depuis le retriever)
    faithfulness = llm_score(FAITHFULNESS_PROMPT, {
        "context": "(context retrieved from Maguette's CV and project documents)",
        "question": question,
        "answer": response,
    })
    relevancy = llm_score(RELEVANCY_PROMPT, {
        "question": question,
        "answer": response,
    })
    completeness = llm_score(COMPLETENESS_PROMPT, {
        "question": question,
        "answer": response,
    })

    print(f"  Faithfulness  : {faithfulness}/10")
    print(f"  Relevancy     : {relevancy}/10")
    print(f"  Completeness  : {completeness}/10")

    # 4. Score global (moyenne des 3 métriques LLM, ramenée sur 10)
    global_score = round((faithfulness + relevancy + completeness) / 3, 2)
    print(f"  ⭐ Score global : {global_score}/10")

    return {
        "question": question,
        "profile": profile,
        "response": response,
        "latency_s": latency,
        "precision_keywords": precision,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "completeness": completeness,
        "global_score": global_score,
    }


# ---------------------------------------------------------------------------
# Pipeline d'évaluation complet
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """Lance l'évaluation sur tout le jeu de test et génère le rapport."""

    print("=" * 55)
    print("  EVALUATION PIPELINE — MaguetteAgent")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    results = []

    for i, test_case in enumerate(TEST_SET):
        # session_id unique par test pour éviter les contaminations d'historique
        session_id = f"eval_session_{i}"
        result = evaluate_single(test_case, session_id)
        results.append(result)

    # -----------------------------------------------------------------------
    # Résumé global
    # -----------------------------------------------------------------------
    n = len(results)
    avg_latency     = round(sum(r["latency_s"] for r in results) / n, 3)
    avg_precision   = round(sum(r["precision_keywords"] for r in results) / n, 2)
    avg_faithfulness= round(sum(r["faithfulness"] for r in results) / n, 2)
    avg_relevancy   = round(sum(r["relevancy"] for r in results) / n, 2)
    avg_completeness= round(sum(r["completeness"] for r in results) / n, 2)
    avg_global      = round(sum(r["global_score"] for r in results) / n, 2)

    print(f"\n{'=' * 55}")
    print("  RÉSUMÉ GLOBAL")
    print(f"{'=' * 55}")
    print(f"  Nb questions évaluées : {n}")
    print(f"  Latence moyenne       : {avg_latency}s")
    print(f"  Précision moyenne     : {avg_precision:.0%}")
    print(f"  Faithfulness moy.     : {avg_faithfulness}/10")
    print(f"  Relevancy moy.        : {avg_relevancy}/10")
    print(f"  Completeness moy.     : {avg_completeness}/10")
    print(f"  ⭐ Score global moy.  : {avg_global}/10")
    print(f"{'=' * 55}\n")

    # -----------------------------------------------------------------------
    # Sauvegarde du rapport JSON
    # -----------------------------------------------------------------------
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.REPORT_DIR / f"eval_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "model": "see config.py",
        "judge_model": config.JUDGE_MODEL,
        "nb_questions": n,
        "summary": {
            "avg_latency_s": avg_latency,
            "avg_precision": avg_precision,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "avg_completeness": avg_completeness,
            "avg_global_score": avg_global,
        },
        "details": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 Rapport sauvegardé : {report_path.resolve()}")
    print("✅ Évaluation terminée !")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()