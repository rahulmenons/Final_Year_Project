import os
import time
import uuid
import json
from django.conf import settings

# --- Core Dependencies for Gemini ---
import google.generativeai as genai

# --- Document Parsing Dependencies (Local ML/Parsing) ---
import fitz  # PyMuPDF
from docx import Document as DocxDocument

# --- Keyword Extraction Dependencies (Local ML) ---
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#------------------------------------
# Chucking function
#------------------------------------

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i+size]))

    return chunks

# Indexing document into vectors

def index_document(document, text):
    chunks = chunk_text(text)

    for chunk in chunks:
        emb = _embedding_model.encode(chunk).tolist()

        DocumentChunk.objects.create(
            document=document,
            text=chunk,
            embedding=emb,
        )
 #----------------------------
 # Retrieve relevant chunks for a query
 #----------------------------
def retrieve_chunks(query, top_k=5):
    q_emb = _embedding_model.encode(query)

    scored = []

    for c in DocumentChunk.objects.all():
        emb = np.array(c.embedding)

        # cosine similarity
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))

        scored.append((sim, c.text))

    scored.sort(reverse=True)

    return [t for _, t in scored[:top_k]]




from .models import CompanyCapability, Document, RFPEvaluation, DocumentChunk
import numpy as np

class DocumentParser:
    @staticmethod
    def parse_pdf(file_obj):
        try:
            file_obj.seek(0)
            file_content = file_obj.read()
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error parsing PDF: {str(e)}")

    @staticmethod
    def parse_docx(file_obj):
        try:
            file_obj.seek(0)
            doc = DocxDocument(file_obj)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Error parsing DOCX: {str(e)}")

    @staticmethod
    def parse_txt(file_obj):
        try:
            file_obj.seek(0)
            text = file_obj.read().decode("utf-8")
            return text.strip()
        except Exception as e:
            raise Exception(f"Error parsing TXT: {str(e)}")

    @classmethod
    def parse(cls, file_obj, file_type):
        file_type_lower = file_type.lower()
        if file_type_lower == "pdf":
            return cls.parse_pdf(file_obj)
        elif file_type_lower in ["docx", "doc"]:
            return cls.parse_docx(file_obj)
        elif file_type_lower == "txt":
            return cls.parse_txt(file_obj)
        else:
            raise ValueError(f"Unsupported file type for local parsing: {file_type}")


# ----------------------------
# KeywordExtractor (unchanged)
# ----------------------------
class KeywordExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeywordExtractor, cls).__new__(cls)
            print("Loading KeyBERT model... (this may take a minute)")
            cls._instance.model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
            print("KeyBERT model loaded successfully!")
        return cls._instance

    def extract_keywords(self, text, top_n=15, ngram_range=(1, 2)):
        if not text or len(text.strip()) < 50:
            return []
        try:
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=top_n,
                use_maxsum=True,
                diversity=0.7,
            )
            return keywords
        except Exception as e:
            raise Exception(f"Error extracting keywords: {str(e)}")

# ----------------------------
# DocumentSummarizer (unchanged except GEMINI key use)
# ----------------------------
class DocumentSummarizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            api_key = "API_KEY_HERE"
            if not api_key:
                print("GEMINI_API_KEY not set; summaries will fail.")
                cls._instance.model = None
            else:
                genai.configure(api_key=api_key)
                cls._instance.model = genai.GenerativeModel("models/gemini-2.5-flash")
        return cls._instance

    def generate_summary(self, text, max_length=250):
        if not self.model or not text or len(text.strip()) < 100:
            return "Text too short or Gemini not configured."

        prompt = (
            f"Create a detailed, professional summary (~{max_length} words) of this document.\n"
            "Focus on main objectives, key methods, important results, and conclusions.\n\n"
            "Document:\n"
            f"{text}\n\nSummary:"
        )

        resp = self.model.generate_content(prompt)
        return resp.text.strip()


# ----------------------------
# RFPMetadataExtractor (RAG VERSION)
# ----------------------------

class RFPMetadataExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            api_key = "API_KEY_HERE"

            if not api_key:
                cls._instance.model = None
            else:
                genai.configure(api_key=api_key)
                cls._instance.model = genai.GenerativeModel("models/gemini-2.5-flash")

        return cls._instance

    def extract_metadata(self, query_hint="budget timeline emd team size") -> dict:

        if not self.model:
            return {
                "budget_in_inr": None,
                "emd_in_inr": None,
                "timeline_weeks": None,
                "no_of_days_for_analysis": None,
                "no_of_days_for_submission": None,
                "confidence": "low",
                "notes": "Gemini not configured.",
            }

        # 🔥 RAG retrieval
        budget_ctx = retrieve_chunks("project budget estimated cost", 3)
        timeline_ctx = retrieve_chunks("project timeline milestones days weeks schedule", 3)
        emd_ctx = retrieve_chunks("earnest money deposit emd", 2)
        team_ctx = retrieve_chunks("team size staffing manpower professionals", 2)
        no_days_submission_ctx = retrieve_chunks("days for submission", 2)
        no_days_analysis_ctx = retrieve_chunks("days for analysis", 2)
        chunks = budget_ctx + timeline_ctx + emd_ctx + team_ctx + no_days_submission_ctx + no_days_analysis_ctx


        if not chunks:
            return {
                "budget_in_inr": None,
                "emd_in_inr": None,
                "timeline_weeks": None,
                "no_of_days_for_analysis": None,
                "no_of_days_for_submission": None,
            
                "confidence": "low",
                "notes": "No relevant chunks retrieved.",
            }

        context = "\n\n".join(chunks)

        prompt = f"""
Extract structured RFP info using ONLY this context.

Return JSON ONLY:

{{
  "budget_in_inr": <int or null>,
  "emd_in_inr": <int or null>,
  "timeline_weeks": <int or null>,
  "no_of_days_for_analysis": <int or null>,
  "no_of_days_for_submission": <int or null>,
  "confidence": "<high|medium|low>",
  "notes": "<short explanation>"
}}

Context:
{context}
"""

        raw = self.model.generate_content(prompt).text.strip()

        try:
            js = raw[raw.find("{"):raw.rfind("}")+1]
            data = json.loads(js)
        except Exception:
            return {
                "budget_in_inr": None,
                "emd_in_inr": None,
                "timeline_weeks": None,
                "no_of_days_for_analysis": None,
                "no_of_days_for_submission": None,
                "confidence": "low",
                "notes": "JSON parse failed",
            }

        def safe(v):
            try:
                s = str(v).lower().replace(",", "").replace("₹", "")
                if "lakh" in s:
                    return int(float("".join(c for c in s if c.isdigit() or c==".")) * 100000)
                if "crore" in s:
                    return int(float("".join(c for c in s if c.isdigit() or c==".")) * 10000000)
                return int(float("".join(c for c in s if c.isdigit() or c==".")))
            except:
                return None

        result = {
            "budget_in_inr": safe(data.get("budget_in_inr")),
            "emd_in_inr": safe(data.get("emd_in_inr")),
            "timeline_weeks": safe(data.get("timeline_weeks")),
            "no_of_days_for_analysis": safe(data.get("no_of_days_for_analysis")),
            "no_of_days_for_submission": safe(data.get("no_of_days_for_submission")),
            "confidence": data.get("confidence", "low"),
            "notes": data.get("notes", ""),
        }

        print("\n🎯 RAG METADATA:", result)
        return result

# ----------------------------
# Scoring helpers (CLEAN VERSION)
# ----------------------------

def _get_company_capability() -> CompanyCapability:
    """
    Make sure we have exactly one CompanyCapability row.
    """
    cap = CompanyCapability.objects.first()
    if not cap:
        raise RuntimeError("CompanyCapability is not configured in the database.")
    return cap


def _safe_int(value, default=0):
    """
    Convert any value (None, str, float) to int safely.
    """
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

# ----------------------------
# computing technical fit by comparing embeddings to keywords.
def _compute_technical_fit(document: Document, cap: CompanyCapability) -> float:
    """
    Semantic technical fit using embeddings.
    0–100 scale.
    """
    if not cap.tech_keywords:
        return 0.0

    company_text = " ".join(cap.tech_keywords)
    company_emb = _embedding_model.encode(company_text)

    sims = []

    for c in DocumentChunk.objects.filter(document=document):
        emb = np.array(c.embedding)

        sim = np.dot(company_emb, emb) / (
            np.linalg.norm(company_emb) * np.linalg.norm(emb)
        )

        sims.append(sim)

    if not sims:
        return 0.0

    return round(max(sims) * 100, 2)


def _compute_budget_fit(rfp_budget: int, cap: CompanyCapability) -> float:
    """
    0–100 measure: is the RFP budget inside our comfortable range?
    """
    rfp_budget = _safe_int(rfp_budget, 0)
    if rfp_budget <= 0:
        return 0.0

    # within range = perfect
    if cap.min_budget <= rfp_budget <= cap.max_budget:
        return 100.0

    # below our min → proportional penalty
    if rfp_budget < cap.min_budget and cap.min_budget > 0:
        return round(rfp_budget / cap.min_budget * 100, 2)

    # above our max → too large for us
    return 0.0


# ----------------------------
# computing complexity based on size and chunk count
def _compute_complexity(document: Document) -> float:
    """
    Rough project complexity based on document size + chunk count.
    0–100 scale.
    """
    text_len = len(document.content_preview or "")
    chunk_count = DocumentChunk.objects.filter(document=document).count()

    complexity = (text_len / 1000) + (chunk_count * 2)

    return min(100.0, round(complexity, 2))

# ----------------------------
# computing compliance based on keywords

def _compute_compliance(document: Document) -> float:
    """
    Estimates compliance / regulatory burden from keywords.
    0–100 scale.
    """
    keywords = [
        "cert-in",
        "iso",
        "audit",
        "security",
        "government",
        "hosting",
        "penetration",
        "meity",
        "nic",
    ]

    text = (document.content_preview or "").lower()

    hits = sum(1 for k in keywords if k in text)

    return min(100.0, hits * 15)

# ----------------------------
# computes timeline pressure but checking it with the complexity

def _compute_timeline_pressure(rfp_timeline_weeks, complexity):
    """
    Measures deadline pressure vs project complexity.
    Higher = more risky.
    """
    if not rfp_timeline_weeks or rfp_timeline_weeks <= 0:
        return 0.0

    pressure = complexity / rfp_timeline_weeks

    return min(100.0, round(pressure * 10, 2))

def _compute_timeline_fit(rfp_timeline_weeks: int, cap: CompanyCapability) -> float:
    """
    0–100 measure: is the RFP timeline realistic for us?
    """

    rfp_timeline_weeks = _safe_int(rfp_timeline_weeks, 0)

    if rfp_timeline_weeks <= 0:
        return 0.0

    if cap.min_timeline_weeks <= rfp_timeline_weeks <= cap.max_timeline_weeks:
        return 100.0

    # shorter than our minimum → proportional penalty
    if rfp_timeline_weeks < cap.min_timeline_weeks and cap.min_timeline_weeks > 0:
        return round(rfp_timeline_weeks / cap.min_timeline_weeks * 100, 2)

    # longer than our max → subtract 5 points per extra week
    extra = rfp_timeline_weeks - cap.max_timeline_weeks
    return max(0.0, 100.0 - extra * 5)




def evaluate_and_save(document: Document) -> RFPEvaluation:
    """
    Enterprise-style RFP evaluation:

    STEP 1: Hard gates (instant reject)
    STEP 2: Core capability score
    STEP 3: Risk score
    STEP 4: Final score = Core - Risk*0.3
    STEP 5: Decision bands
    """

    cap = _get_company_capability()

    # --------------------
    # Load extracted RFP values
    # --------------------

    meta = getattr(document, "rfp_metadata", None) or {}

    rfp_budget = _safe_int(document.rfp_budget or meta.get("budget_in_inr"), 0)
    rfp_timeline = _safe_int(document.rfp_timeline_weeks or meta.get("timeline_weeks"), 0)

    # --------------------
    # Compute base metrics
    # --------------------

    technical = _compute_technical_fit(document, cap)
    budget = _compute_budget_fit(rfp_budget, cap)
    timeline = _compute_timeline_fit(rfp_timeline, cap)

    complexity = _compute_complexity(document)
    compliance = _compute_compliance(document)
    timeline_pressure = _compute_timeline_pressure(rfp_timeline, complexity)

    # ======================================================
    # STEP 1 — HARD GATES
    # ======================================================

    hard_fail_reasons = []

    if budget == 0:
        hard_fail_reasons.append("Budget outside acceptable range")

    if timeline == 0:
        hard_fail_reasons.append("Timeline not feasible")

    if technical < 40:
        hard_fail_reasons.append("Technical capability too low")

    if hard_fail_reasons:
        reasoning = "Hard rejection: " + "; ".join(hard_fail_reasons)

        evaluation, _ = RFPEvaluation.objects.update_or_create(
            document=document,
            defaults={
                "technical_fit_score": technical,
                "budget_fit_score": budget,
                "timeline_fit_score": timeline,
                "overall_fit_score": 0,
                "decision": "REJECT",
                "reasoning": reasoning,
            },
        )

        document.status = "REJECTED"
        document.processed = True
        document.save(update_fields=["status", "processed"])

        return evaluation

    # ======================================================
    # STEP 2 — CORE CAPABILITY SCORE
    # ======================================================

    core_fit = round(
        technical * 0.4 +
        budget * 0.3 +
        timeline * 0.3,
        2
    )

    # ======================================================
    # STEP 3 — RISK SCORE
    # ======================================================

    risk = round(
        timeline_pressure * 0.4 +
        compliance * 0.3 +
        complexity * 0.3,
        2
    )

    # ======================================================
    # STEP 4 — FINAL SCORE
    # ======================================================

    final_score = round(core_fit - (risk * 0.3), 2)

    # ======================================================
    # STEP 5 — DECISION
    # ======================================================

    if final_score >= 70:
        decision = "ACCEPT"
        doc_status = "ACCEPTED"
    elif final_score >= 50:
        decision = "REVIEW"
        doc_status = "REVIEW"
    else:
        decision = "REJECT"
        doc_status = "REJECTED"

    reasoning = (
        f"CoreFit={core_fit}% | Risk={risk}% | Final={final_score}% | "
        f"Technical={technical}% | Budget={budget}% | Timeline={timeline}% | "
        f"Complexity={complexity}% | Compliance={compliance}% | Pressure={timeline_pressure}%"
    )

    evaluation, _ = RFPEvaluation.objects.update_or_create(
        document=document,
        defaults={
            "technical_fit_score": technical,
            "budget_fit_score": budget,
            "timeline_fit_score": timeline,
            "overall_fit_score": final_score,
            "decision": decision,
            "reasoning": reasoning,
        },
    )
    
    document.status = doc_status
    document.processed = True
    document.save(update_fields=["status", "processed"])

    return evaluation
