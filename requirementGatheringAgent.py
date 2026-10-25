import os
import sys
import json
from html import escape
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    import weasyprint
except Exception:
    weasyprint = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv(Path(__file__).resolve().parent / ".env")

# ==========================================
# 1. DEFINE THE STRUCTURED OUTPUT SCHEMAS
# ==========================================

class UserStory(BaseModel):
    id: str = Field(description="Unique identifier for the user story, e.g., US-001")
    title: str = Field(description="Short descriptive title of the feature.")
    as_a: str = Field(description="The user persona or role.")
    i_want_to: str = Field(description="The specific action or capability desired.")
    so_that: str = Field(description="The business value or reason for the feature.")
    acceptance_criteria: List[str] = Field(description="Explicit Given/When/Then or checklist conditions for completeness.")
    priority: str = Field(description="Calculated priority level: High, Medium, or Low based on text context.")

class FeatureModule(BaseModel):
    module_name: str = Field(description="Name of the functional component or module (e.g., Authentication, Checkout).")
    user_stories: List[UserStory] = Field(description="List of user stories belonging to this specific module.")

class StructuredRequirements(BaseModel):
    project_name: str = Field(description="Extracted or inferred name of the project.")
    summary: str = Field(description="High level overview of the requirement document.")
    modules: List[FeatureModule] = Field(description="Grouped business features extracted from the document.")

# ==========================================
# 2. DEFINE THE LANGGRAPH AGENT STATE
# ==========================================

class QAProjectState(TypedDict):
    pdf_path: str                             # Input: Path to the PDF requirement doc
    raw_extracted_text: str                   # Intermediate: Raw text parsed from PDF
    refined_features: Dict[str, Any]          # Output: Structured Pydantic response parsed into a dict
    output_pdf_report: str
    user_stories_json_path: str

# ==========================================
# 3. DEFINE THE AGENT FUNCTIONS (NODES)
# ==========================================

def pdf_parsing_node(state: QAProjectState) -> Dict[str, Any]:
    """Reads the local PDF file and extracts text page by page."""
    pdf_path = state["pdf_path"]
    print(f"🔄 Parsing PDF Document: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Requirement document not found at: {pdf_path}")
        
    reader = PdfReader(pdf_path)
    full_text = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
            
    print(f"✅ Successfully parsed {len(reader.pages)} pages.")
    return {"raw_extracted_text": "\n".join(full_text)}


def llm_requirement_processor_node(state: QAProjectState) -> Dict[str, Any]:
    """Sends raw text to the OpenAI LLM and maps it to the StructuredRequirements schema."""
    print("🧠 Invoking LLM for User Story generation...")
    
    # 1. Initialize the LLM (Using a model with high reasoning capacity like gpt-4o)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # 2. Bind the structured output schema to the model
    structured_llm = llm.with_structured_output(StructuredRequirements)
    
    # 3. Create a strict prompt engineering template for user stories
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Principal Business Analyst and QA Director.\n"
            "Your job is to read raw product requirement documents (PRD) and translate them into actionable, "
            "production-grade User Stories for engineering and QA teams.\n\n"
            "Strictly follow the requested output JSON structure. Ensure that every User Story contains explicit, "
            "testable Acceptance Criteria."
        )),
        ("user", "Here is the raw text extracted from the requirements document:\n\n{document_text}")
    ])
    
    # 4. Chain the prompt and the structured LLM execution together
    chain = prompt | structured_llm
    
    # 5. Run the chain using text stored in the graph state
    result: StructuredRequirements = chain.invoke({"document_text": state["raw_extracted_text"]})
    
    # Convert Pydantic object back into standard dictionary format to update state
    return {"refined_features": result.model_dump()}


def user_story_pdf_generation_node(state: QAProjectState) -> Dict[str, Any]:
    """Save structured user stories as JSON and render a readable PDF report."""
    data = state["refined_features"]
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "user_stories.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)

    html_parts = [
        """<!doctype html>
        <html><head><meta charset="utf-8"><style>
        @page { size: A4; margin: 20mm; }
        body { font-family: sans-serif; color: #263238; line-height: 1.5; }
        h1 { color: #174a6e; border-bottom: 2px solid #2878a5; padding-bottom: 8px; }
        h2 { color: #2878a5; margin-top: 28px; border-bottom: 1px solid #ccd7dc; }
        .story { border: 1px solid #d5dfe3; padding: 12px; margin: 14px 0; break-inside: avoid; }
        .story h3 { margin: 0 0 8px; }
        .priority { color: #52636b; font-size: 10pt; }
        </style></head><body>
        <h1>User Story Requirements</h1>""",
        f"<p><strong>Project:</strong> {escape(str(data.get('project_name', 'Unnamed project')))}</p>",
        f"<p><strong>Summary:</strong> {escape(str(data.get('summary', '')))}</p>",
    ]

    for module in data.get("modules", []):
        html_parts.append(f"<h2>{escape(str(module.get('module_name', 'Module')))}</h2>")
        for story in module.get("user_stories", []):
            html_parts.append(
                f"<section class='story'><h3>{escape(str(story.get('id', '')))}: "
                f"{escape(str(story.get('title', 'Untitled story')))}</h3>"
                f"<p class='priority'><strong>Priority:</strong> {escape(str(story.get('priority', 'Unspecified')))}</p>"
                f"<p><strong>As a</strong> {escape(str(story.get('as_a', '')))}, "
                f"<strong>I want to</strong> {escape(str(story.get('i_want_to', '')))}, "
                f"<strong>so that</strong> {escape(str(story.get('so_that', '')))}.</p>"
                "<p><strong>Acceptance criteria</strong></p><ul>"
            )
            html_parts.extend(
                f"<li>{escape(str(criterion))}</li>"
                for criterion in story.get("acceptance_criteria", [])
            )
            html_parts.append("</ul></section>")

    html_parts.append("</body></html>")
    pdf_path = output_dir / "user_stories_document.pdf"

    if weasyprint is not None:
        weasyprint.HTML(string="".join(html_parts)).write_pdf(str(pdf_path))
    else:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        import re

        text = "".join(html_parts)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>|</li>|</h[1-6]>|</section>|</div>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = [Paragraph("User Story Requirements", styles["Title"]), Spacer(1, 16)]
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                story.append(Paragraph(cleaned, styles["BodyText"]))
        doc.build(story)

    print(f"📄 User story PDF generated: {pdf_path}")

    return {
        "output_pdf_report": str(pdf_path),
        "user_stories_json_path": str(json_path),
    }

# ==========================================
# 4. ORCHESTRATE VIA LANGGRAPH
# ==========================================

# Initialize the StateGraph with the state dictionary layout
workflow = StateGraph(QAProjectState)

# Add our processing units as graph nodes
workflow.add_node("parse_pdf", pdf_parsing_node)
workflow.add_node("generate_user_stories", llm_requirement_processor_node)
workflow.add_node("generate_user_story_report", user_story_pdf_generation_node)

# Set up the sequential pipeline connections
workflow.set_entry_point("parse_pdf")
workflow.add_edge("parse_pdf", "generate_user_stories")
workflow.add_edge("generate_user_stories", "generate_user_story_report")
workflow.add_edge("generate_user_story_report", END)

# Compile into an executable application agent
requirement_agent = workflow.compile()

# ==========================================
# 5. RUN EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    default_pdf_path = root_dir / "generated" / "prd_document.pdf"

    pdf_path = default_pdf_path
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1]).expanduser().resolve()
    elif not default_pdf_path.exists():
        raise FileNotFoundError(
            f"No PRD PDF was found at {default_pdf_path}. Run prdGenerator.py first or pass a PDF path explicitly."
        )

    initial_input = {"pdf_path": str(pdf_path)}

    try:
        final_output_state = requirement_agent.invoke(initial_input)

        print("\n================ GENERATED FRAMEWORK STATE OUT: ================")
        print(json.dumps(final_output_state["refined_features"], indent=2))
        print(f"\nUser stories PDF: {final_output_state['output_pdf_report']}")
        print(f"User stories JSON: {final_output_state['user_stories_json_path']}")

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        print("Please verify the PDF path and set OPENAI_API_KEY before running the agent.")
