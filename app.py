import os
import re
import json
import streamlit as st
from pypdf import PdfReader
from typing import List
from groq import Groq
from tavily import TavilyClient

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruthLayer | AI Fact-Check Agent",
    page_icon="🔍",
    layout="wide"
)

# --- API INITIALIZATION ---
def get_keys():
    groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")
    return groq_key, tavily_key

def get_active_groq_model(groq_client: Groq) -> str:
    """Finds the first working chat model on your Groq account."""
    models = groq_client.models.list().data
    blocked = ["whisper", "guard", "tts", "embed", "vision", "orpheus", "canopylabs", "playai"]
    available_ids = [
        m.id for m in models
        if not any(b in m.id.lower() for b in blocked)
    ]
    preferred = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3.8-27b",
        "minimaxai/minimax-m2.7",
        "meta-models/muse-spark-open",
    ]
    ordered = [p for p in preferred if p in available_ids] + [m for m in available_ids if m not in preferred]

    for candidate in ordered:
        try:
            groq_client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return candidate
        except Exception:
            continue
    raise RuntimeError("No working chat model found on this Groq key.")

def extract_json_safely(raw_text: str) -> dict:
    """Strips any thinking tags or markdown fences and parses JSON cleanly."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)

# --- CORE PIPELINE FUNCTIONS ---
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def extract_claims(groq_client: Groq, model_name: str, text: str) -> List[str]:
    prompt = f"""
    You are an expert investigative fact-checker. Analyze the following document text and extract all specific, verifiable factual claims.
    Focus strictly on:
    - Statistics, percentages, and metrics
    - Financial figures, valuations, and revenue numbers
    - Dates, timelines, and historical milestones
    - Specific technical specs or named entity achievements

    Ignore subjective opinions, promotional slogans, and general advice. Extract up to 15 of the most critical factual claims.

    Respond ONLY with a valid JSON object in this exact format:
    {{"claims": ["claim 1", "claim 2"]}}

    Document Text:
    {text[:15000]}
    """
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a precise fact extraction assistant. Return raw valid JSON only, with no extra text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    data = extract_json_safely(response.choices[0].message.content)
    return data.get("claims", [])

def verify_claim(groq_client: Groq, model_name: str, tavily_client: TavilyClient, claim: str) -> dict:
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

    Respond ONLY with a valid JSON object in this exact format:
    {{
      "status": "Verified",
      "explanation": "Concise 1-2 sentence explanation comparing the PDF claim against live web evidence.",
      "real_fact": "The accurate, up-to-date fact or statistic found in the web search results.",
      "source_url": "The most authoritative URL from the search results supporting this verdict, or 'None found' if False."
    }}
    """
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a strict fact-checking judge. Return raw valid JSON only, with no extra text."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.0,
    )
    verdict = extract_json_safely(response.choices[0].message.content)
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
            active_model = get_active_groq_model(groq_client)

            with st.status(f"Analyzing document using `{active_model}`...", expanded=True) as status:
                st.write("📄 Extracting text from PDF...")
                raw_text = extract_text_from_pdf(uploaded_file)
                
                if not raw_text:
                    st.error("Could not extract text from this PDF. Ensure it is not a scanned image-only PDF.")
                    st.stop()

                st.write("🧠 Identifying verifiable claims (stats, dates, figures)...")
                claims = extract_claims(groq_client, active_model, raw_text)
                st.write(f"Found **{len(claims)}** verifiable claims.")

                if not claims:
                    st.warning("No specific statistical or factual claims were detected in this PDF.")
                    st.stop()

                results = []
                progress_bar = st.progress(0)
                for idx, claim in enumerate(claims):
                    st.write(f"🌐 Verifying ({idx + 1}/{len(claims)}): *{claim[:80]}...*")
                    verdict = verify_claim(groq_client, active_model, tavily_client, claim)
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
