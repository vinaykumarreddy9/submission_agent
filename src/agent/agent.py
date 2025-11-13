from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    job_description: str
    candidate_profile: str
    mail_body: Optional[str] = None


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

JOB_ANALYSIS_ENGINE_PROMPT = PromptTemplate.from_template(
    """
    **ROLE: Senior Talent Acquisition Partner and Candidate Pitch Specialist**

    **TASK:** Draft a complete, professional, and highly concise email to a Recruiter. The email's sole purpose is to present the **top 3-5 most compelling, evidence-based alignments** that make the candidate an exceptional fit for the role.

    **INPUTS:**
    1.  JOB_DESCRIPTION: {job_description}
    2.  CANDIDATE_PROFILE: {candidate_profile}

    **STRICT CONSTRAINTS (MANDATORY ADHERENCE):**
    1.  **Advocacy Focus:** The tone must be professionally persuasive, focusing *only* on the candidate's strengths relative to the job requirements.
    2.  **Conciseness:** The email body must be extremely brief. Limit the core alignment points to a maximum of five (5) bullet points. Avoid all unnecessary introductory or transitional sentences.
    3.  **Data Integrity:** All claims MUST be directly traceable to the provided CANDIDATE_PROFILE. **DO NOT infer, assume, or fabricate** any skills or experience.
    4.  **Tone:** The entire email must be professional, polite, and formal.

    **OUTPUT FORMAT (MANDATORY EMAIL STRUCTURE):**
    Generate the complete email using the following structure. Use the placeholders exactly as shown.

    ---
    **Subject:** High-Priority Candidate Review: Strong Alignment for [Insert Job Title from JD]

    Dear Recruiter,

    Please review this candidate profile. Based on a direct analysis against the job description, the following 3-5 points highlight why this individual is a strong, immediate fit for the role:

    *   [**Alignment Point 1:** State the job requirement, followed by the direct, high-impact evidence/metric from the profile.]
    *   [**Alignment Point 2:** Direct, high-impact alignment point.]
    *   [**Alignment Point 3:** Direct, high-impact alignment point.]
    *   [**Alignment Point 4 (Optional):** Highly relevant secondary skill or experience.]
    *   [**Alignment Point 5 (Optional):** Highly relevant secondary skill or experience.]

    I recommend moving forward with this profile.

    Best regards,
    Team NinjaHire
    ---
"""
)

def submission_agent(state : AgentState) -> AgentState:
    """Analyzes the job description and candidate profile and returns a e-mail."""
    job_description = state.get("job_description", "")
    candidate_profile = state.get("candidate_profile","")

    if not job_description or not candidate_profile:
        raise("job description or candidate_profile cannot be empty.")
    
    prompt = JOB_ANALYSIS_ENGINE_PROMPT.format(candidate_profile=candidate_profile,job_description=job_description)
    response = model.invoke(prompt)
    return {
        "mail_body" : response.content
    }


graph = (
    StateGraph(AgentState)
    .add_node("submission_agent", submission_agent)
    .add_edge("__start__", "submission_agent")
    .add_edge("submission_agent", END)
    .compile(name="recruitment_agent")
)
