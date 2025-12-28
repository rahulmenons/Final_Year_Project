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

from .models import CompanyCapability, Document, RFPEvaluation

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
            api_key = "AIzaSyALEE6TVeuZW8iHfqb0ximU75mm9v4BTfU"
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
# RFPMetadataExtractor (UPDATED)
# ----------------------------
class RFPMetadataExtractor:
    """
    Uses Gemini to extract structured fields (budget, timeline, team size, EMD, submission/analysis days)
    from unstructured RFP text. Logs the extracted JSON to the server console.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            api_key = "AIzaSyALEE6TVeuZW8iHfqb0ximU75mm9v4BTfU"
            if not api_key:
                print("GEMINI_API_KEY not set; RFP metadata extraction will fail.")
                cls._instance.model = None
            else:
                genai.configure(api_key=api_key)
                cls._instance.model = genai.GenerativeModel("models/gemini-2.5-flash")
        return cls._instance

    def extract_metadata(self, text: str) -> dict:
        """
        Returns:
        {
          "budget_in_inr": int|null,
          "emd_in_inr": int|null,
          "timeline_weeks": int|null,
          "no_of_days_for_analysis": int|null,
          "no_of_days_for_submission": int|null,
          "team_size_required": int|null,
          "confidence": "high|medium|low",
          "notes": "..."
        }
        """
        if not self.model or not text or len(text.strip()) < 50:
            result = {
                "budget_in_inr": None,
                "emd_in_inr": None,
                "timeline_weeks": None,
                "no_of_days_for_analysis": None,
                "no_of_days_for_submission": None,
                "team_size_required": None,
                "confidence": "low",
                "notes": "Not enough text or model not configured.",
            }
            # Log immediately
            print("\n🔍 RFP metadata extraction (skipped) — not enough text or model not set.")
            print(result)
            return result

        prompt = (
            "You are an assistant that extracts structured information from RFP (Request for Proposal) documents.\n\n"
            "Read the following RFP text and extract:\n"
            "- project budget (convert to a single integer in INR, e.g. 800000 for ₹8,00,000),\n"
            "- earnest Money Deposit (EMD) if mentioned (convert to integer in INR, or null if not mentioned),\n"
            "- overall project timeline in weeks,\n"
            "- no. of days for analysis of the project timeline,\n"
            "- no. of days for submission of detailed project,\n"
            "- approximate team size required.\n\n"
            "If something is not explicitly mentioned, make your BEST REASONABLE GUESS based on the context,\n"
            "but mark confidence as \"low\", \"medium\", or \"high\".\n\n"
            "RESPOND WITH JSON ONLY in this exact format:\n\n"
            '{\n'
            '  "budget_in_inr": <integer or null>,\n'
            '  "emd_in_inr": <integer or null>,\n'
            '  "timeline_weeks": <integer or null>,\n'
            '  "no_of_days_for_analysis": <integer or null>,\n'
            '  "no_of_days_for_submission": <integer or null>,\n'
            '  "team_size_required": <integer or null>,\n'
            '  "confidence": "<high|medium|low>",\n'
            '  "notes": "<short explanation>"\n'
            '}\n\n'
            "Do NOT include any extra text outside the JSON.\n\n"
            "RFP TEXT:\n"
        )

        full_prompt = prompt + "\n" + text[:15000]  # truncate to avoid huge prompts

        # Call Gemini
        resp = self.model.generate_content(full_prompt)
        raw = resp.text.strip()

        # Try to parse JSON robustly
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            # Attempt to extract JSON substring
            try:
                json_start = raw.find("{")
                json_end = raw.rfind("}") + 1
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_sub = raw[json_start:json_end]
                    data = json.loads(json_sub)
                else:
                    data = None
            except Exception:
                data = None

        if not data:
            # Last-resort: try to find key-like pairs with simple heuristics (very permissive)
            print("\n⚠️ Warning: Failed to parse JSON directly from Gemini response.")
            print("Raw response preview (first 1000 chars):")
            print(raw[:1000])
            # Return a low-confidence result
            result = {
                "budget_in_inr": None,
                "emd_in_inr": None,
                "timeline_weeks": None,
                "no_of_days_for_analysis": None,
                "no_of_days_for_submission": None,
                "team_size_required": None,
                "confidence": "low",
                "notes": "Failed to parse JSON from model response.",
            }
            print("\n🔍 RFP metadata (FAILED to parse):")
            print(result)
            return result

        # Normalize ints
        def to_int_or_none(v):
            try:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return int(v)
                # allow strings like "800000", "8,00,000", "5 lakhs"
                s = str(v).strip()
                # remove commas and currency symbols and words
                s = s.replace(",", "")
                s = s.replace("₹", "")
                s = s.lower().replace("inr", "").strip()
                # handle lakhs/crores
                if "lakh" in s or "lac" in s:
                    # convert "5 lakh" -> 500000
                    num = float("".join(ch for ch in s if (ch.isdigit() or ch == ".")))
                    return int(num * 100000)
                if "crore" in s:
                    num = float("".join(ch for ch in s if (ch.isdigit() or ch == ".")))
                    return int(num * 10000000)
                # strip non-digit
                cleaned = ""
                for ch in s:
                    if ch.isdigit() or ch == ".":
                        cleaned += ch
                if cleaned == "":
                    return None
                return int(float(cleaned))
            except Exception:
                return None

        result = {
            "budget_in_inr": to_int_or_none(data.get("budget_in_inr")),
            "emd_in_inr": to_int_or_none(data.get("emd_in_inr") or data.get("emd") or data.get("emd_in_inr")),
            "timeline_weeks": to_int_or_none(data.get("timeline_weeks")),
            "no_of_days_for_analysis": to_int_or_none(data.get("no_of_days_for_analysis")),
            "no_of_days_for_submission": to_int_or_none(data.get("no_of_days_for_submission")),
            "team_size_required": to_int_or_none(data.get("team_size_required")),
            "confidence": data.get("confidence", "low"),
            "notes": data.get("notes", ""),
        }

        # Pretty log the extraction results to the server console
        try:
            print("\n" + "=" * 40)
            print("🎯 EXTRACTED RFP METADATA (from Gemini)")
            print("=" * 40)
            print(f"Budget (INR): {result['budget_in_inr']}")
            print(f"EMD (INR): {result['emd_in_inr']}")
            print(f"Timeline (weeks): {result['timeline_weeks']}")
            print(f"Days for analysis: {result['no_of_days_for_analysis']}")
            print(f"Days for submission: {result['no_of_days_for_submission']}")
            print(f"Team size required: {result['team_size_required']}")
            print(f"Confidence: {result['confidence']}")
            if result.get("notes"):
                print(f"Notes: {result['notes']}")
            print("=" * 40 + "\n")
        except Exception:
            # fallback simple print
            print("Extracted metadata:", result)

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


def _compute_technical_fit(document: Document, cap: CompanyCapability) -> float:
    """
    How well RFP keywords match company tech_keywords (0–100).
    """
    # keywords linked to this document
    doc_keywords = set(document.keywords.values_list("keyword__keyword", flat=True))
    doc_keywords = {k.lower().strip() for k in doc_keywords if k}

    # company capability tech stack
    company_keywords = {k.lower().strip() for k in (cap.tech_keywords or []) if k}

    if not doc_keywords or not company_keywords:
        return 0.0

    overlap = doc_keywords.intersection(company_keywords)
    # fraction of company skills that appear in RFP
    return round(len(overlap) / len(company_keywords) * 100, 2)


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


def _compute_capacity_fit(team_size_required: int, cap: CompanyCapability) -> float:
    """
    0–100 measure: does required team size fit within our max_team_size?
    """
    team_size_required = _safe_int(team_size_required, 0)
    if team_size_required <= 0:
        return 0.0

    if team_size_required <= cap.max_team_size:
        return 100.0

    # if they want more than our max, give a ratio score
    return round(cap.max_team_size / team_size_required * 100, 2)


def evaluate_and_save(document: Document) -> RFPEvaluation:
    """
    Main scoring function:
    - Reads Document + CompanyCapability + keywords
    - Computes scores
    - Saves/updates RFPEvaluation
    - Updates Document.status and processed flag
    """
    cap = _get_company_capability()

    # Prefer explicit rfp_* fields. If missing, try rfp_metadata JSON.
    meta = getattr(document, "rfp_metadata", None) or {}
    rfp_budget = document.rfp_budget or meta.get("budget_in_inr") or 0
    rfp_timeline = document.rfp_timeline_weeks or meta.get("timeline_weeks") or 0
    rfp_team = document.rfp_team_size_required or meta.get("team_size_required") or 0

    rfp_budget = _safe_int(rfp_budget, 0)
    rfp_timeline = _safe_int(rfp_timeline, 0)
    rfp_team = _safe_int(rfp_team, 0)

    technical = _compute_technical_fit(document, cap)
    budget = _compute_budget_fit(rfp_budget, cap)
    timeline = _compute_timeline_fit(rfp_timeline, cap)
    capacity = _compute_capacity_fit(rfp_team, cap)

    # Weights – tweak later if needed
    w_tech, w_budget, w_timeline, w_capacity = 0.3, 0.4, 0.2, 0.1
    overall = round(
        technical * w_tech +
        budget * w_budget +
        timeline * w_timeline +
        capacity * w_capacity,
        2,
    )

    # Decision rules
    if overall < 60 or capacity < 40 or budget == 0.0:
        decision = "REJECT"
        doc_status = "REJECTED"
    elif overall >= 80:
        decision = "ACCEPT"
        doc_status = "ACCEPTED"
    else:
        decision = "REVIEW"
        doc_status = "REVIEW"

    reasoning = (
        f"Technical fit: {technical}% | "
        f"Budget fit: {budget}% | "
        f"Timeline fit: {timeline}% | "
        f"Capacity fit: {capacity}% | "
        f"Overall: {overall}% → Decision: {decision}"
    )

    evaluation, _ = RFPEvaluation.objects.update_or_create(
        document=document,
        defaults={
            "technical_fit_score": technical,
            "budget_fit_score": budget,
            "timeline_fit_score": timeline,
            "capacity_fit_score": capacity,
            "overall_fit_score": overall,
            "decision": decision,
            "reasoning": reasoning,
        },
    )

    # Update document status + mark processed
    document.status = doc_status
    document.processed = True
    document.save(update_fields=["status", "processed"])

    return evaluation
