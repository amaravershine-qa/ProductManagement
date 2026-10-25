import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import weasyprint
except Exception:
    weasyprint = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Load local environment variable configurations (.env)
load_dotenv(Path(__file__).resolve().parent / ".env")

# ==========================================
# 1. STRUCTURAL SCHEMAS (Pydantic Models)
# ==========================================

class FunctionalRequirement(BaseModel):
    requirement_id: str = Field(description="Unique requirement code, e.g., FR-001")
    title: str = Field(description="Short, action-oriented title of the feature function.")
    description: str = Field(description="Deep functional description of how the system must behave.")
    target_persona: str = Field(description="The primary user category interacting with this requirement.")
    impact: str = Field(description="Business priority or system critical impact rating (Critical, High, Medium, Low).")

class OutOfScopeItem(BaseModel):
    title: str = Field(description="Feature or area excluded from this iteration phase.")
    rationale: str = Field(description="Clear business or technical justification for exclusion.")

class StructuredPRD(BaseModel):
    product_name: str = Field(description="Finalized formal name engineered for the product application.")
    executive_summary: str = Field(description="High-level operational overview, mission statement, and problem context.")
    target_audience: List[str] = Field(description="Identified user personas, demographics, or primary target system profiles.")
    core_objectives: List[str] = Field(description="Key key performance metrics, success vectors, or operational milestones.")
    functional_requirements: List[FunctionalRequirement] = Field(description="Comprehensive catalog of baseline system features.")
    out_of_scope: List[OutOfScopeItem] = Field(description="Explicitly identified boundaries and excluded target features.")
    technical_constraints: List[str] = Field(description="Architectural considerations, regulatory mandates, security, or platform expectations.")

# ==========================================
# 2. FRAMEWORK AGENT STATE DEFINITION
# ==========================================

class PRDGenerationState(TypedDict):
    user_idea_input: str
    structured_prd_data: Dict[str, Any]
    output_prd_pdf_path: str

# ==========================================
# 3. CORE PROCESSING NODES
# ==========================================

def llm_prd_generator_node(state: PRDGenerationState) -> Dict[str, Any]:
    """Leverages a reasoning LLM to extrapolate a raw concept into a structured product engineering specification."""
    print("🧠 Node 1: Brainstorming system specs and structuring the PRD payload data...")
    user_concept = state["user_idea_input"]

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    structured_llm = llm.with_structured_output(StructuredPRD)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an Elite Principal Product Manager and Technical Software Architect.\n"
            "Your objective is to ingest primitive, high-level product notes or feature descriptions from a user, "
            "and expand them into a thoroughly fleshed-out, professional-grade Product Requirement Document (PRD).\n\n"
            "Requirements for your expansion:\n"
            "- Extract or invent a professional product title if none is explicitly present.\n"
            "- Detail explicit, production-grade functional requirements with clear IDs (FR-001, FR-002, etc.).\n"
            "- Be highly technical, providing clear constraints around authentication, security, scale, and operational assumptions.\n"
            "- Map out concrete out-of-scope parameters to protect engineering timelines."
        )),
        ("user", "Flesh out this basic idea into a comprehensive, structured technical PRD document:\n\n{idea}")
    ])

    chain = prompt | structured_llm
    result: StructuredPRD = chain.invoke({"idea": user_concept})
    return {"structured_prd_data": result.model_dump()}


def prd_pdf_compilation_node(state: PRDGenerationState) -> Dict[str, Any]:
    """Compiles the structured PRD into a clean PDF without exposing any raw HTML content."""
    print("🎨 Node 2: Compiling professional A4 specification layout for Product Design...")
    prd = state["structured_prd_data"]

    def build_plain_prd_text() -> str:
        from datetime import date

        lines: List[str] = []
        lines.append(prd.get("product_name", "System Specification Project"))
        lines.append("")
        lines.append("Product Requirement Document (PRD)")
        lines.append(f"Document ID: PRD-{date.today().strftime('%Y%m%d')}")
        lines.append(f"Version: 1.0")
        lines.append(f"Date: {date.today().strftime('%d-%b-%Y')}")
        lines.append(f"Status: Draft")
        lines.append(f"Target Audience: {', '.join(prd.get('target_audience', []))}")
        lines.append("")

        lines.append("1. Overview")
        lines.append(prd.get("executive_summary", "No overview summary provided."))
        lines.append("")
        lines.append("Problem Statement:")
        lines.append("The product must address the operational inefficiencies faced by its target users by consolidating essential workflows into a single structured platform and reducing manual effort, information gaps, and delays.")
        lines.append("")

        lines.append("2. Objectives")
        for objective in prd.get("core_objectives", []):
            lines.append(f"- {objective}")
        lines.append("")
        lines.append("Success Measures:")
        lines.append("- Improved productivity and reduced operational delays")
        lines.append("- Higher user satisfaction and adoption among core stakeholders")
        lines.append("- Better quality, consistency, and timeliness of service delivery")
        lines.append("")

        lines.append("3. Scope")
        lines.append("In Scope:")
        for fr in prd.get("functional_requirements", []):
            lines.append(f"- {fr['requirement_id']} - {fr['title']} ({fr['impact']})")
        lines.append("")
        lines.append("Out of Scope:")
        for scope_item in prd.get("out_of_scope", []):
            lines.append(f"- {scope_item['title']}: {scope_item['rationale']}")
        lines.append("")

        lines.append("4. Users and Personas")
        for persona in prd.get("target_audience", []):
            lines.append(f"- {persona}")
        lines.append("")

        lines.append("5. Functional Requirements")
        for fr in prd.get("functional_requirements", []):
            lines.append(f"{fr['requirement_id']} - {fr['title']} ({fr['impact']})")
            lines.append(f"Target Persona: {fr['target_persona']}")
            lines.append(f"Description: {fr.get('description', '')}")
            lines.append("Business Value: The feature provides measurable value by improving operational efficiency, reducing errors, and supporting the user workflow.")
            lines.append("")

        lines.append("6. Constraints")
        for constraint in prd.get("technical_constraints", []):
            lines.append(f"- {constraint}")
        lines.append("")
        lines.append("Dependencies:")
        lines.append("- Access to required data sources and system integrations")
        lines.append("- Available technical infrastructure and deployment environment")
        lines.append("- Stakeholder review and timely feedback during implementation")
        lines.append("")

        lines.append("7. Risks and Assumptions")
        lines.append("- Risks: integration complexity, data quality issues, security and compliance requirements, schedule constraints, and change in stakeholder priorities.")
        lines.append("- Assumptions: required data will be available, users will participate in validation activities, and the platform will be implemented within the agreed delivery timeline.")
        lines.append("")
        lines.append("Open Questions:")
        lines.append("- Final list of integrations to be included in the initial release")
        lines.append("- Compliance requirements and approved hosting environment")
        lines.append("- User training and rollout timeline")
        lines.append("")

        lines.append("8. Approval")
        lines.append("Prepared by: Product Team")
        lines.append("Reviewed by: Business Stakeholders")
        lines.append("Approved by: __________________________")
        lines.append("Date: ________________________________")

        return "\n".join(lines)

    plain_text = build_plain_prd_text()

    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_output_path = output_dir / "prd_document.pdf"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=12,
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            spaceAfter=6,
        )

        doc = SimpleDocTemplate(str(target_output_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = [Paragraph("Product Requirement Document", title_style), Spacer(1, 8)]

        for paragraph in plain_text.split("\n"):
            cleaned = paragraph.strip()
            if cleaned:
                story.append(Paragraph(cleaned, body_style))

        doc.build(story)
    except Exception:
        if weasyprint is not None:
            try:
                html_content = """
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body { font-family: "Times New Roman", serif; font-size: 11pt; line-height: 1.5; margin: 20mm; }
                        p { margin: 0 0 8pt 0; }
                    </style>
                </head>
                <body>
                    <pre style="font-family: 'Times New Roman', serif; white-space: pre-wrap; word-wrap: break-word;">{plain_text}</pre>
                </body>
                </html>
                """.format(plain_text=plain_text)
                weasyprint.HTML(string=html_content).write_pdf(str(target_output_path))
            except Exception:
                raise
        else:
            raise

    print(f"📄 PRD PDF generated: {target_output_path}")
    return {"output_prd_pdf_path": str(target_output_path)}

# ==========================================
# 4. ORCHESTRATE WITH LANGGRAPH
# ==========================================

workflow = StateGraph(PRDGenerationState)
workflow.add_node("generate_prd", llm_prd_generator_node)
workflow.add_node("compile_prd_pdf", prd_pdf_compilation_node)
workflow.set_entry_point("generate_prd")
workflow.add_edge("generate_prd", "compile_prd_pdf")
workflow.add_edge("compile_prd_pdf", END)

prd_agent = workflow.compile()

# ==========================================
# 5. RUN EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        product_name = " ".join(sys.argv[1:]).strip()
    else:
        product_name = input("Enter the product name for the PRD: ").strip()

    if not product_name:
        product_name = "Custom Product"

    user_idea = (
        f"Create a Product Requirement Document (PRD) for the product named '{product_name}'. "
        f"Include the product vision, target users, business goals, key features, out-of-scope items, "
        f"and technical constraints."
    )

    try:
        final_output_state = prd_agent.invoke({"user_idea_input": user_idea})
        print("\n================ GENERATED PRD STATE OUT: ================")
        print(json.dumps(final_output_state["structured_prd_data"], indent=2))
        print(f"\nPRD PDF: {final_output_state['output_prd_pdf_path']}")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        print("Please verify your OPENAI_API_KEY before running the agent.")
