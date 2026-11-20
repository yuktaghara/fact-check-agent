# 🔍 TruthLayer: Automated PDF Fact-Check Agent

An AI-powered web application that acts as a **Truth Layer** for marketing documents, whitepapers, and reports. Users upload a PDF, and the agent automatically extracts factual claims, cross-references them against live web search results, and flags outdated or fabricated statistics.

## 🏗️ Architecture & Workflow
1. **PDF Extraction (`pypdf`):** Parses raw text from uploaded PDF documents.
2. **Claim Extraction (`Gemini 2.5 Flash` + `Pydantic`):** Uses structured JSON output schemas to isolate objective, verifiable claims (statistics, dates, financial figures, technical specs) while ignoring subjective marketing copy.
3. **Live Web Verification (`Tavily Search API`):** Executes real-time web searches for each claim to retrieve authoritative, up-to-date evidence.
4. **Fact-Checking Judge (`Gemini 2.5 Flash`):** Evaluates the original claim against live web snippets and classifies each item as:
   - ✅ **Verified:** Matches live web data.
   - ⚠️ **Inaccurate:** Contains outdated statistics, wrong dates, or distorted figures.
   - 🚨 **False:** Completely fabricated or unsupported by credible sources.

## 🚀 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd fact-check-agent