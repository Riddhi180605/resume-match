# 📄 ResumeMatch Pro — AI Resume Analyzer

**ResumeMatch Pro** is a web-based **AI-powered resume & job description matching tool** that analyzes a candidate’s resume against a given job description and provides:

- 🎯 A **percentage match score**
- 🧩 **Missing skills & improvement suggestions**
- 📑 A **downloadable PDF analysis report**

The system uses **Google Gemini (Generative AI)** with **schema-enforced JSON output**, robust error handling, and a smart fallback algorithm to ensure reliability.

---

## 🚀 Features

- 📤 Upload resumes in **PDF or DOCX**
- 📝 Paste any job description
- 🤖 AI-based resume–JD comparison (Gemini 2.5 Flash)
- 📊 Matching score (0–100%)
- 🧠 Skill gap & improvement suggestions
- 🛡 Schema-validated AI responses (JSON enforced)
- ⚠ Automatic fallback analysis if AI fails
- 📄 Downloadable PDF report
- 🎨 Modern glassmorphism UI
- 🔐 Secure environment variable configuration

---

## 🧠 How It Works

1. **Resume Upload**
   - Supports `.pdf` and `.docx`
   - Extracts text using `PyPDF2` and `python-docx`

2. **Text Processing**
   - Large resumes are **chunked safely** to respect model limits
   - Key content is combined with job description

3. **AI Analysis**
   - Uses **Gemini 2.5 Flash**
   - Enforces strict JSON schema:
     ```json
     {
       "matching_score": 0-100,
       "suggested_improvements": ["skill1", "skill2", ...]
     }
     ```

4. **Fallback Engine**
   - Keyword overlap scoring if AI response fails
   - Ensures the app never breaks

5. **PDF Report**
   - Generates a structured report using `reportlab`
   - Includes score, suggestions, and text snippets

---

## 🛠 Tech Stack

### Backend
- **Python**
- **Flask**
- **Google Gemini (google-genai)**
- **ReportLab (PDF generation)**
- **PyPDF2**
- **python-docx**
- **dotenv**

### Frontend
- **HTML5**
- **CSS3 (Glassmorphism UI)**
- **Jinja2 Templates**

---

## 📂 Project Structure

resumematch-pro/
│
├── app.py # Flask application & AI logic
├── requirements.txt # Python dependencies
├── .env # API keys 
│
├── templates/
│ └── index.html # UI template
│
├── static/
│ └── style.css # styling
│
└── README.md # Project documentation