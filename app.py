import os
import json
import streamlit as st
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import List, Literal
from groq import Groq
from tavily import TavilyClient

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruthLayer | AI Fact-Check Agent",
    page_icon="🔍",
    layout="wide"
)

# --- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT ---
class ExtractedClaims(BaseModel):
    claims: List[str] = Field(
        description="List of specific, objective, verifiable factual claims (statistics, dates, financial figures, historical/technical facts). Exclude opinions, marketing fluff, or vague statements."
    )

class VerificationVerdict(BaseModel):
    status: Literal["Verified", "Inaccurate", "False"] = Field(
        description="'Verified' if live search confirms the claim; 'Inaccurate' if the claim has outdated numbers, wrong dates, or partial distortions; 'False' if completely fabricated or contradicted with no supporting evidence."
    )
    explanation: str = Field(
        description="Concise 1-2 sentence explanation comparing the PDF claim against live web evidence."
    )
    real_fact: str = Field(
        description="The accurate, up-to-date fact or statistic found in the web search results. If Verified, restate the confirmed fact."
    )
    source_url: str = Field(
        description="The most authoritative URL from the search results supporting this verdict, or 'None found' if False."
    )

# --- API INITIALIZATION ---
def get_keys():
    groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")
    return groq_key, tavily_key

# --- CORE PIPELINE FUNCTIONS ---
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def extract_claims(groq_client: Groq, text: str) -> List[str]:
    prompt = f"""
    You are an expert investigative fact-checker. Analyze the following document text and extract all specific, verifiable factual claims.
    Focus strictly on:
    - Statistics, percentages, and metrics
    - Financial figures, valuations, and revenue numbers
    - Dates, timelines, and historical milestones
    - Specific technical specs or named entity achievements

    Ignore subjective opinions, promotional slogans, and general advice. Extract up to 15 of the most critical factual claims.

    Return ONLY valid JSON matching this exact schema:
    {{"claims": ["claim 1", "claim 2"]}}

    Document Text:
    {text[:15000]}
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a precise JSON-outputting fact extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("claims", [])

def verify_claim(groq_client: Groq, tavily_client: TavilyClient, claim: str) -> dict:
    # 1. Search Live Web
    search_response = tavily_client.search(
        query=claim,
        search_depth="advanced",
        max_results=4
    )
    results = search_response.get("results", [])
    
    formatted_evidence = "\n\n".join(
        [f"Source: {r.get('url')}\nTitle: {r.get('title')}\nSnippet: {r.get('content')}" for r in results]
    )

    # 2. Cross-reference with LLM
    judge_prompt = f"""
    You are a strict Fact-Checking Judge. Compare the Claim extracted from a PDF against the Live Web Search Evidence.
    Note: Current year is 2026. Pay close attention to whether statistics or figures in the claim are outdated compared to the latest available data.

    CLAIM FROM PDF:
    "{claim}"

    LIVE WEB SEARCH EVIDENCE:
    {formatted_evidence if formatted_evidence else "No relevant web results found."}

    Classify the claim strictly into one of three statuses:
    - "Verified": The live web evidence directly confirms the numbers, dates, and facts in the claim.
    - "Inaccurate": The claim is based on a real topic/entity, but the statistic, date, or figure is outdated, exaggerated, or wrong.
    - "False": The claim is completely fabricated, contradicts consensus facts, or has zero credible evidence supporting it.

    Return ONLY valid JSON matching this exact schema:
    {{
      "status": "Verified" | "Inaccurate" | "False",
      "explanation": "Concise 1-2 sentence explanation comparing the PDF claim against live web evidence.",
      "real_fact": "The accurate, up-to-date fact or statistic found in the web search results.",
      "source_url": "The most authoritative URL from the search results supporting this verdict, or 'None found' if False."
    }}
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a strict fact-checking judge that outputs valid JSON only."},
            {"role": "user", "content": judge_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    verdict = json.loads(response.choices[0].message.content)
    verdict["claim"] = claim
    return verdict

# --- UI LAYOUT ---
st.title("🔍 TruthLayer: Automated PDF Fact-Check Agent")
st.markdown(
    "Upload any marketing PDF, report, or whitepaper. The agent extracts factual claims, cross-references them against live web data, and flags outdated or hallucinated stats."
)

groq_key, tavily_key = get_keys()

with st.sidebar:
    st.header("⚙️ Configuration")
    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded")
        
    if not tavily_key:
        tavily_key = st.text_input("Tavily API Key", type="password")
    else:
        st.success("Tavily API Key loaded")

    st.divider()
    st.markdown("**Classification Legend:**")
    st.markdown("✅ **Verified:** Matches live web data")
    st.markdown("⚠️ **Inaccurate:** Outdated stats or wrong figures")
    st.markdown("🚨 **False:** Fabricated or no evidence found")

uploaded_file = st.file_uploader("Upload a PDF Document for Verification", type=["pdf"])

if uploaded_file is not None:
    if not groq_key or not tavily_key:
        st.error("Please provide both Groq and Tavily API keys in the sidebar or Streamlit Secrets.")
        st.stop()

    if st.button("🚀 Run Automated Fact-Check", type="primary"):
        try:
            groq_client = Groq(api_key=groq_key)
            tavily_client = TavilyClient(api_key=tavily_key)

            with st.status("Analyzing document...", expanded=True) as status:
                st.write("📄 Extracting text from PDF...")
                raw_text = extract_text_from_pdf(uploaded_file)
                
                if not raw_text:
                    st.error("Could not extract text from this PDF. Ensure it is not a scanned image-only PDF.")
                    st.stop()

                st.write("🧠 Identifying verifiable claims (stats, dates, figures)...")
                claims = extract_claims(groq_client, raw_text)
                st.write(f"Found **{len(claims)}** verifiable claims.")

                if not claims:
                    st.warning("No specific statistical or factual claims were detected in this PDF.")
                    st.stop()

                results = []
                progress_bar = st.progress(0)
                for idx, claim in enumerate(claims):
                    st.write(f"🌐 Verifying ({idx + 1}/{len(claims)}): *{claim[:80]}...*")
                    verdict = verify_claim(groq_client, tavily_client, claim)
                    results.append(verdict)
                    progress_bar.progress((idx + 1) / len(claims))

                status.update(label="Fact-Check Complete!", state="complete", expanded=False)

            # --- SUMMARY METRICS ---
            verified_count = sum(1 for r in results if r["status"] == "Verified")
            inaccurate_count = sum(1 for r in results if r["status"] == "Inaccurate")
            false_count = sum(1 for r in results if r["status"] == "False")

            st.divider()
            st.subheader("📊 Audit Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims Checked", len(results))
            col2.metric("✅ Verified", verified_count)
            col3.metric("⚠️ Inaccurate / Outdated", inaccurate_count)
            col4.metric("🚨 False / Hallucinated", false_count)

            st.divider()
            st.subheader("📋 Detailed Claim-by-Claim Report")

            priority_order = {"False": 0, "Inaccurate": 1, "Verified": 2}
            sorted_results = sorted(results, key=lambda x: priority_order.get(x["status"], 3))

            for item in sorted_results:
                status_label = item["status"]
                with st.container(border=True):
                    if status_label == "Verified":
                        st.success(f"✅ **VERIFIED:** {item['claim']}")
                    elif status_label == "Inaccurate":
                        st.warning(f"⚠️ **INACCURATE / OUTDATED:** {item['claim']}")
                    else:
                        st.error(f"🚨 **FALSE / NO EVIDENCE:** {item['claim']}")

                    st.markdown(f"**Analysis:** {item['explanation']}")
                    if status_label != "Verified":
                        st.info(f"**Correct / Real Fact:** {item['real_fact']}")
                    if item.get("source_url") and item["source_url"] != "None found":
                        st.caption(f"🔗 Source: [{item['source_url']}]({item['source_url']})")

        except Exception as e:
            st.error(f"API Error: {str(e)}")
