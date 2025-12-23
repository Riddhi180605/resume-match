import os
import json
import uuid
import tempfile
import logging
from datetime import datetime
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from dotenv import load_dotenv
import docx
import PyPDF2
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")  # change in prod

# Initialize Gemini client
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in .env")
client = genai.Client(api_key=API_KEY)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_CHUNK_CHARS = 25000  # tune as necessary per model limits (safety buffer)


# ------------------------
# Helper functions
# ------------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def extract_text_from_file(path: str) -> str:
    text = ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            doc = docx.Document(path)
            for p in doc.paragraphs:
                if p.text:
                    text += p.text + "\n"
        elif ext == ".pdf":
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            logger.warning("Unsupported extension for extraction: %s", ext)
    except Exception as e:
        logger.exception("Failed to extract text: %s", e)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS):
    """Yield chunks of text not exceeding max_chars (prefer splitting on paragraphs)."""
    if len(text) <= max_chars:
        yield text
        return

    paragraphs = text.split("\n\n")
    chunk = ""
    for p in paragraphs:
        # if single paragraph is huge, slice it
        if len(p) > max_chars:
            # flush current chunk first
            if chunk:
                yield chunk
                chunk = ""
            # slice paragraph
            for i in range(0, len(p), max_chars):
                yield p[i:i + max_chars]
        else:
            if len(chunk) + len(p) + 2 <= max_chars:
                chunk += (p + "\n\n")
            else:
                yield chunk
                chunk = p + "\n\n"
    if chunk:
        yield chunk


def safe_generate_json(prompt: str):
    """
    Call Gemini with schema enforcement and robust parsing.
    Returns dict on success, None on failure.
    """
    # define a permissive schema (score as number)
    json_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "matching_score": types.Schema(
                type=types.Type.NUMBER,
                description="Numeric percentage match from 0 to 100."
            ),
            "suggested_improvements": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="List of 3-5 missing/weak key skills."
            )
        },
        required=["matching_score", "suggested_improvements"]
    )

    system_instruction = (
        "You are an expert HR recruiter and resume analyzer. "
        "Compare the provided candidate resume text against the job description. "
        "Return **only** valid JSON matching the schema: matching_score (0-100 number) "
        "and suggested_improvements (array of 3-5 strings). Be concise."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )

        # The SDK often returns a structured object. Try multiple ways to get text.
        raw_text = None
        # 1) response.text (some SDKs)
        raw_text = getattr(response, "text", None)
        if not raw_text:
            # 2) candidates path
            try:
                raw_text = response.candidates[0].content[0].text
            except Exception:
                pass
        if not raw_text:
            try:
                raw_text = json.dumps(response)  # last resort - may not be JSON
            except Exception:
                raw_text = None

        if not raw_text:
            logger.error("No text returned from model response object.")
            return None

        # Clean and parse JSON
        # Sometimes there are stray characters, attempt to find first { ... }
        text = raw_text.strip()
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_text = text[first:last + 1]
        else:
            json_text = text

        parsed = json.loads(json_text)
        # normalize
        score = parsed.get("matching_score")
        suggestions = parsed.get("suggested_improvements", [])
        return {"matching_score": float(score), "suggested_improvements": suggestions}
    except Exception as e:
        logger.exception("LLM generate/parsing error: %s", e)
        return None


def simple_fallback_analysis(resume: str, jd: str):
    """
    Lightweight fallback: counts overlapping keywords and returns a basic score + suggestions.
    This is NOT as smart as LLM but keeps the app functional.
    """
    def words(s):
        tokens = re_split_nonword.split(s.lower())
        return [t for t in tokens if t and len(t) > 1]

    import re
    re_split_nonword = re.compile(r"[^a-z0-9\+\#\.]+", re.I)

    resume_words = set(words(resume))
    jd_words = set(words(jd))
    overlap = resume_words.intersection(jd_words)
    # compute simple proportional score
    if len(jd_words) == 0:
        score = 0.0
    else:
        score = (len(overlap) / len(jd_words)) * 100
    # suggest top 5 JD words not in resume
    missing = list(jd_words - resume_words)
    # rank by frequency in JD
    from collections import Counter
    jd_counts = Counter(words(jd))
    missing_sorted = sorted(missing, key=lambda w: -jd_counts[w])
    suggestions = missing_sorted[:5]
    return {"matching_score": round(score, 1), "suggested_improvements": suggestions}


def create_pdf_report(score: float, suggestions: list, resume_text: str, jd_text: str, out_path: str):
    c = canvas.Canvas(out_path, pagesize=letter)
    w, h = letter
    x = 40
    y = h - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "ResumeMatch Pro - Analysis Report")
    c.setFont("Helvetica", 12)
    y -= 30
    c.drawString(x, y, f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    y -= 25
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"Matching Score: {score}%")
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Suggested Improvements / Missing Skills:")
    y -= 18
    c.setFont("Helvetica", 11)
    for s in suggestions:
        for line in split_text_for_pdf(s, 80):
            if y < 80:
                c.showPage()
                y = h - 40
            c.drawString(x + 10, y, f"- {line}")
            y -= 14
    # Optionally include short snippets of resume/JD (truncated)
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    if y < 120:
        c.showPage()
        y = h - 40
    c.drawString(x, y, "Resume snippet:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in split_text_for_pdf(resume_text[:3000], 100):
        if y < 80:
            c.showPage()
            y = h - 40
        c.drawString(x, y, line)
        y -= 12
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    if y < 120:
        c.showPage()
        y = h - 40
    c.drawString(x, y, "Job Description snippet:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in split_text_for_pdf(jd_text[:3000], 100):
        if y < 80:
            c.showPage()
            y = h - 40
        c.drawString(x, y, line)
        y -= 12

    c.save()


def split_text_for_pdf(text, width_chars=80):
    # naive splitter by characters
    lines = []
    i = 0
    while i < len(text):
        lines.append(text[i:i + width_chars])
        i += width_chars
    return lines

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    report_filename = None

    if request.method == "POST":
        jd_text = request.form.get("job_description", "").strip()
        resume_file = request.files.get("resume_file")

        # Basic validations
        if not jd_text:
            flash("Please paste the job description.", "danger")
            return redirect(url_for("index"))
        if not resume_file or resume_file.filename == "":
            flash("Please upload a resume file (PDF or DOCX).", "danger")
            return redirect(url_for("index"))
        if not allowed_file(resume_file.filename):
            flash("Unsupported file type. Use PDF or DOCX.", "danger")
            return redirect(url_for("index"))

        # Save uploaded file to a secure temp file
        ext = os.path.splitext(resume_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_name = tmp.name
            resume_file.save(temp_name)

        # Extract text, then remove temp file
        resume_text = extract_text_from_file(temp_name)
        try:
            os.remove(temp_name)
        except Exception:
            pass
        if not resume_text:
            flash("Could not extract text from the uploaded resume. If it's a scanned PDF, OCR is required (not included).", "warning")
            # still allow analysis with empty text (fallback)
        
        # Build a prompt from chunks (combine resume and JD in manageable size)
        # We'll feed the model a single combined prompt but truncate/merge intelligently
        # If resume_text is too long, take the first N chars from each chunk (or summarize chunks first)
        combined_jd = jd_text.strip()
        combined_resume = resume_text.strip()

        # If resume is huge, only use first X chars per chunk or use the fallback
        if len(combined_resume) > (MAX_CHUNK_CHARS * 3):
            # reduce to first 3 chunks to be safe
            chunks = list(chunk_text(combined_resume, MAX_CHUNK_CHARS))
            combined_resume = "\n\n".join(chunks[:3])

        # Prepare prompt
        prompt = f"""
Compare the resume and job description and produce JSON strictly in this format:
{{"matching_score": number_between_0_and_100, "suggested_improvements": ["skill1", "skill2", ...]}}

Resume:
{combined_resume}

Job Description:
{combined_jd}
"""
        # Call LLM
        llm_result = safe_generate_json(prompt)
        # If LLM failed, fallback
        if not llm_result:
            logger.info("LLM failed, using fallback analyzer.")
            result = simple_fallback_analysis(combined_resume, combined_jd)
            used_fallback = True
        else:
            result = llm_result
            used_fallback = False
        # Format results
        score = round(result.get("matching_score", 0), 1)
        suggestions = result.get("suggested_improvements", [])
        if not suggestions:
            suggestions = ["No specific gaps found — resume appears to match well."]

        # Create report PDF
        unique_id = uuid.uuid4().hex
        report_filename = f"resume_report_{unique_id}.pdf"
        out_path = os.path.join(tempfile.gettempdir(), report_filename)
        try:
            create_pdf_report(score, suggestions, combined_resume, combined_jd, out_path)
        except Exception as e:
            logger.exception("Failed to create PDF: %s", e)
            report_filename = None
        results = {
            "score": f"{score}%",
            "suggestions": suggestions,
            "used_fallback": used_fallback,
            "report_name": report_filename
        }
    return render_template("index.html", results=results)

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    safe_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(safe_path):
        flash("Report not found or expired.", "danger")
        return redirect(url_for("index"))
    return send_file(safe_path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    # ensure temp dir exists
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))