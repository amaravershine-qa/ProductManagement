import json
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

load_dotenv(Path(__file__).resolve().parent / ".env")

# ==========================================
# 1. STRUCTURAL SCHEMAS (Pydantic Models)
# ==========================================

class TestCase(BaseModel):
    test_id: str = Field(description="Unique tracking ID for the test case, e.g., TC-US001-01")
    title: str = Field(description="Short summary of the scenario being verified.")
    test_type: str = Field(description="Type of test (e.g., Positive, Negative, Boundary, Edge Case).")
    pre_conditions: List[str] = Field(description="Setup prerequisites required before executing the test.")
    steps: List[str] = Field(description="Ordered test actions in plain-language step sequence, not Gherkin syntax.")
    expected_result: str = Field(description="The explicit verification criteria or successful end state.")

class ModuleTestCases(BaseModel):
    module_name: str = Field(description="Name of the Epic or module matching the input context.")
    test_cases: List[TestCase] = Field(description="List of test scenarios mapping to this module.")

class StructuredTestSuite(BaseModel):
    project_name: str = Field(description="Name of the application target being tested.")
    modules: List[ModuleTestCases] = Field(description="Grouped test suits organized by system components.")

# ==========================================
# 2. FRAMEWORK AGENT STATE DEFINITION
# ==========================================

class QATestGenState(TypedDict):
    refined_features: Dict[str, Any]          # Input: Structural user stories dict passed from Agent 1
    test_cases_data: Dict[str, Any]          # Intermediate: Structural JSON payload containing test cases
    output_pdf_report: str                    # Output: Path to the generated PDF test cases artifact

# ==========================================
# 3. CORE PROCESSING NODES
# ==========================================

def llm_test_generator_node(state: QATestGenState) -> Dict[str, Any]:
    """Passes user story details to the LLM to design comprehensive functional test suites."""
    print("🧠 Node 1: Designing test scenarios and translating criteria into Plain Language sequences...")
    
    # 1. Capture the input user stories structured dictionary
    user_stories_json = state["refined_features"]
    
    # 2. Initialize reasoning LLM configuration
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    structured_llm = llm.with_structured_output(StructuredTestSuite)
    
    # 3. Structure prompt template instructing the LLM on solid QA coverage practices
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Lead QA Engineer and Test Automation Architect.\n"
            "Your job is to read user stories along with their acceptance criteria, and engineer an "
            "exhaustive functional test suite.\n\n"
            "Guidelines:\n"
            "- For each user story, write multiple test scenarios covering Positive paths, Negative validation inputs, and vital Edge Cases.\n"
            "- Write the execution 'steps' as a plain-language sequence of numbered actions, not Gherkin syntax.\n"
            "- Keep the summary short and descriptive so it fits a spreadsheet-style 'Test Case Summary' column.\n"
            "- Ensure the pre_conditions are real-world actionable setups (e.g., User is logged in, item is in cart).\n"
            "- Format the output as spreadsheet-friendly rows with columns: Serial Number, Test Case Summary, Test Case Steps, Expected Result."
        )),
        ("user", "Convert the following structured user story specifications into a robust test suite:\n\n{stories_data}")
    ])
    
    # 4. Chain execution and return clean mapped output dict
    chain = prompt | structured_llm
    result: StructuredTestSuite = chain.invoke({"stories_data": str(user_stories_json)})
    
    return {"test_cases_data": result.model_dump()}


def _normalize_test_case_row(test_case: Any) -> Dict[str, Any]:
    """Convert any LLM/test case payload into a flat spreadsheet row."""
    if isinstance(test_case, str):
        try:
            test_case = json.loads(test_case)
        except json.JSONDecodeError:
            return {
                "serial_number": "-",
                "summary": test_case,
                "steps": "-",
                "expected_result": "-",
            }

    if not isinstance(test_case, dict):
        return {
            "serial_number": "-",
            "summary": str(test_case),
            "steps": "-",
            "expected_result": "-",
        }

    steps = test_case.get("steps", [])
    if isinstance(steps, list):
        step_text = "<br/>".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    elif isinstance(steps, str):
        step_text = steps.replace("\n", "<br/>")
    else:
        step_text = str(steps)

    summary = test_case.get("summary") or test_case.get("title") or "-"
    return {
        "serial_number": "-",
        "summary": str(summary),
        "steps": step_text,
        "expected_result": str(test_case.get("expected_result", "-")),
    }


def test_pdf_generation_node(state: QATestGenState) -> Dict[str, Any]:
    """Compiles the structured test scenario data objects into an Excel spreadsheet workbook."""
    print("🎨 Node 2: Compiling test case workbook in Excel format...")
    data = state["test_cases_data"]

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openpyxl is required to generate .xlsx files. Install it with: pip install openpyxl"
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "Test Cases"

    headers = ["Serial Number", "Test Case Summary", "Test Case Steps", "Expected Result"]
    ws.append(headers)

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    header_font = Font(bold=True, color="1F2D3D")
    thin = Side(style="thin", color="B9C7D5")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.border = border
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    row_index = 2
    for module in data.get("modules", []):
        for tc in module.get("test_cases", []):
            normalized = _normalize_test_case_row(tc)
            steps = normalized["steps"]
            if not isinstance(steps, str):
                steps = str(steps)
            steps = steps.replace("<br/>", "\n")
            row = [str(row_index - 1), normalized["summary"], steps, normalized["expected_result"]]
            ws.append(row)

            for cell in ws[row_index]:
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=True)

            ws[f"A{row_index}"].alignment = Alignment(horizontal="center", vertical="top")
            row_index += 1

    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 48
    ws.column_dimensions["D"].width = 36

    for row_cells in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row_cells:
            if cell.row == 1:
                continue
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # Save the output file inside local file structure
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_output_path = output_dir / "test_cases_document.xlsx"
    wb.save(target_output_path)

    print(f"🎉 Excel workbook generated at: {target_output_path}")


# ==========================================
# 4. ORCHESTRATE WITH LANGGRAPH
# ==========================================

workflow = StateGraph(QATestGenState)

# Append structural nodes
workflow.add_node("generate_scenarios", llm_test_generator_node)
workflow.add_node("compile_pdf", test_pdf_generation_node)

# Map straightforward connections
workflow.set_entry_point("generate_scenarios")
workflow.add_edge("generate_scenarios", "compile_pdf")
workflow.add_edge("compile_pdf", END)

# Compile active runner instance
test_gen_agent = workflow.compile()

# ==========================================
# 5. RUN USING REQUIREMENT AGENT OUTPUT
# ==========================================
if __name__ == "__main__":
    user_stories_path = Path(__file__).resolve().parent / "generated" / "user_stories.json"
    if not user_stories_path.is_file():
        raise FileNotFoundError(
            f"User stories not found at {user_stories_path}. "
            "Run requirementGatheringAgent.py first."
        )

    with user_stories_path.open(encoding="utf-8") as input_file:
        user_stories = json.load(input_file)

    final_state = test_gen_agent.invoke({"refined_features": user_stories})
    output_path = final_state.get("output_pdf_report") or str(Path(__file__).resolve().parent / "generated" / "test_cases_document.xlsx")
    print(f"✅ Test cases workbook generated: {output_path}")
