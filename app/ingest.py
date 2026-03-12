"""
ingest.py
---------
Charge les documents depuis app/data/ (PDF, Markdown, TXT),
les découpe en chunks, génère les embeddings et sauvegarde
le vectorstore FAISS sur le disque.

Usage:
    python -m app.ingest
    # ou depuis la racine du projet :
    python app/ingest.py
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app.config as config

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> List[Document]:
    """Parcourt data_dir et charge tous les fichiers PDF, MD et TXT.

    Retourne une liste de Documents LangChain avec leur contenu et métadonnées
    (source, nom du fichier, type).
    """
    docs: List[Document] = []
    files = list(data_dir.rglob("*"))  # recherche récursive dans les sous-dossiers

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {data_dir.resolve()}")

    for file_path in files:
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                # PyPDFLoader découpe automatiquement par page
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()

            elif suffix in (".md", ".markdown"):
                # UnstructuredMarkdownLoader respecte la structure Markdown
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()

            elif suffix == ".txt":
                # TextLoader charge le fichier texte brut
                # encoding="utf-8" évite les erreurs sur les caractères spéciaux
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()

            else:
                print(f"  [skip] format non supporté : {file_path.name}")
                continue

            # Ajouter des métadonnées utiles à chaque document
            for doc in loaded:
                doc.metadata["source"] = str(file_path)
                doc.metadata["filename"] = file_path.name
                doc.metadata["filetype"] = suffix

            docs.extend(loaded)
            print(f"  [ok] {file_path.name} — {len(loaded)} page(s)/section(s) chargée(s)")

        except Exception as e:
            print(f"  [erreur] {file_path.name} : {e}")

    print(f"\nTotal : {len(docs)} document(s) chargé(s) depuis {data_dir.resolve()}\n")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    """Découpe les documents en chunks avec RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter essaie de couper dans cet ordre :
    paragraphes → phrases → mots → caractères
    pour préserver le sens autant que possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # séparateurs tentés dans l'ordre
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"Chunking : {len(docs)} doc(s) → {len(chunks)} chunk(s) "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})\n")
    return chunks


# ---------------------------------------------------------------------------
# Embedding & vectorstore
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Génère les embeddings pour chaque chunk et construit le vectorstore FAISS."""
    print(f"Génération des embeddings avec le modèle '{config.EMBEDDING_MODEL_NAME}'...")
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    # FAISS.from_documents appelle embeddings.embed_documents() sur tous les chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Vectorstore construit : {vectorstore.index.ntotal} vecteurs indexés\n")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, save_dir: str) -> None:
    """Sauvegarde le vectorstore FAISS sur le disque."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # crée le dossier si besoin
    vectorstore.save_local(save_dir)
    print(f"Vectorstore sauvegardé dans : {Path(save_dir).resolve()}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def ingest() -> None:
    """Lance le pipeline complet : chargement → chunking → embedding → sauvegarde."""

    print("=" * 60)
    print("INGEST PIPELINE — MaguetteAgent")
    print("=" * 60)

    # 1. Vérifier que le dossier data existe
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {config.DATA_DIR.resolve()} n'existe pas. "
            "Crée-le et ajoute tes documents."
        )

    # 2. Charger les documents
    print(f"\n📂 Chargement des documents depuis {config.DATA_DIR.resolve()}...\n")
    docs = load_documents(config.DATA_DIR)

    if not docs:
        raise ValueError("Aucun document chargé. Vérifie le contenu de app/data/")

    # 3. Découper en chunks
    print("✂️  Découpage en chunks...\n")
    chunks = split_documents(docs)

    # 4. Générer les embeddings et construire le vectorstore
    print("🔢 Embedding & construction du vectorstore...\n")
    vectorstore = build_vectorstore(chunks)

    # 5. Sauvegarder sur le disque
    print("💾 Sauvegarde du vectorstore...\n")
    save_vectorstore(vectorstore, config.VECTORSTORE_DIR)

    print("\n✅ Ingest terminé avec succès !")
    print(f"   → {len(chunks)} chunks indexés")
    print(f"   → Vectorstore : {Path(config.VECTORSTORE_DIR).resolve()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest()