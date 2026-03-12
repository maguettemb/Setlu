## Centralised configuration file 
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "data" 
PROJECTS_DIR = DATA_DIR / "projects"    
VECTORSTORE_DIR = BASE_DIR / "vectorstore"  

EMBEDDING_MODEL_NAME = "text-embedding-3-small" 
MODEL_NAME = "gpt-4.1-mini"    
CHAT_MODEL_TEMPERATURE = 0.0

CHUNK_SIZE = 200  
CHUNK_OVERLAP = 50
MAX_RETRIEVALS = 5  

max_length = 500

## Snowflake configuration ## Check that out 
SCHEMA = "PUBLIC"
DOCSTRINGS_SEARCH_SERVICE = "STREAMLIT_DOCSTRINGS_SEARCH_SERVICE"
PAGES_SEARCH_SERVICE = "STREAMLIT_DOCS_PAGES_SEARCH_SERVICE"
HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
DOCSTRINGS_CONTEXT_LEN = 10
PAGES_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

# ---------------------------------------------------------------------------
# Evaluation Config
# ---------------------------------------------------------------------------

REPORT_DIR = Path("app/eval_reports")   # dossier de sauvegarde des rapports
JUDGE_MODEL = "gpt-4o-mini"             # modèle utilisé pour le scoring