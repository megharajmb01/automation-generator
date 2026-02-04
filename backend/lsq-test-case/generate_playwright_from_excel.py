import os
import pandas as pd
from dotenv import load_dotenv
from time import sleep
from google import genai

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "../../frontend/tests/test-files/test_case_for_4.xlsx"
)

OUTPUT_SPEC = os.path.join(
    BASE_DIR,
    "../../frontend/tests/auto-generated.spec.js"
)

PDF_FILE_PATH = "/home/megharaj/Automation Generator/frontend/tests/test-files/sample.pdf"

MODEL_NAME = "gemini-2.0-flash"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def sanitize_llm_code(code: str) -> str:
    """Remove markdown and garbage from Gemini output"""
    return (
        code.replace("```javascript", "")
            .replace("```js", "")
            .replace("```", "")
            .strip()
    )


def requires_file_upload(steps: list[str]) -> bool:
    keywords = [
        "click to upload",
        "upload pdf",
        "select a pdf",
        "choose pdf",
        "drag and drop"
    ]
    text = " ".join(steps).lower()
    return any(k in text for k in keywords)


def enforce_playwright_rules(code: str) -> str:
    """
    Normalize Gemini output into stable Playwright code
    """

    replacements = {
        "getByText('Upload PDF')":
            "locator('.step-item').filter({ hasText: 'Upload PDF' })",

        ".isVisible()":
            ")).toBeVisible()",

        "PDF Loaded Successfully":
            "Ready to Extract",
    }

    for bad, good in replacements.items():
        code = code.replace(bad, good)

    forbidden = [
        "toHaveURL",
        "waitForNavigation",
        "dragAndDrop",
        "fs.",
        "new File(",
        "DataTransfer"
    ]

    for f in forbidden:
        if f in code:
            raise RuntimeError(
                f"\n❌ Forbidden Playwright pattern (cannot auto-fix):\n{f}"
            )

    return code



# --------------------------------------------------
# STEP 1: READ EXCEL
# --------------------------------------------------

def read_excel(file_path):
    df = pd.read_excel(file_path, sheet_name="Positive_Test_Cases")

    def parse_steps(cell):
        if pd.isna(cell):
            return []
        return [
            step.strip()[3:] if step.strip()[1] == "." else step.strip()
            for step in str(cell).split("\n")
        ]

    df["test_steps"] = df["test_steps"].apply(parse_steps)
    return df.to_dict(orient="records")


# --------------------------------------------------
# STEP 2: GEMINI → PLAYWRIGHT
# --------------------------------------------------

def generate_playwright_test(client, test_case):
    upload_rule = ""
    if requires_file_upload(test_case["test_steps"]):
        upload_rule = f"""
FILE UPLOAD RULE (MANDATORY):
- Use ONLY:
  await page.locator('input[type="file"]').setInputFiles('{PDF_FILE_PATH}');
- Then click:
  await page.getByRole('button', {{ name: 'Upload PDF' }}).click();
- Upload success is verified ONLY by:
  await expect(page.getByText('Ready to Extract')).toBeVisible();
"""

    prompt = f"""
You are a senior Playwright automation engineer.

ABSOLUTE RULES:
- Output ONLY a Playwright test() function
- NO imports
- NO describe()
- NO markdown
- NO code fences
- Assume navigation to PDF page is already done
- Use expect() for all assertions
- NEVER use getByText('Upload PDF')
- NEVER check URL changes
- NEVER use isVisible()
- Prefer getByRole or stable container locators

{upload_rule}

Test Case ID: {test_case["test_case_no"]}
Description: {test_case["test_description"]}

Test Steps:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(test_case["test_steps"]))}

Expected Result:
{test_case["expected_result"]}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    code = sanitize_llm_code(response.text)
    return enforce_playwright_rules(code)


# --------------------------------------------------
# STEP 3: GENERATE SPEC FILE
# --------------------------------------------------

def generate_spec(test_cases):
    client = genai.Client(api_key=os.getenv("API_KEY"))
    tests = []

    for idx, tc in enumerate(test_cases, start=1):
        print(f"🤖 Generating test {idx}/{len(test_cases)}")
        test_code = generate_playwright_test(client, tc)
        tests.append(test_code)
        sleep(1.2)

    final_code = f"""
const {{ test, expect }} = require('@playwright/test');

test.describe('AI Generated Playwright Tests', () => {{

  test.beforeEach(async ({{ page }}) => {{
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', {{ name: 'PDF Test Cases' }}).click();
  }});

{chr(10).join(tests)}

}});
"""

    os.makedirs(os.path.dirname(OUTPUT_SPEC), exist_ok=True)
    with open(OUTPUT_SPEC, "w", encoding="utf-8") as f:
        f.write(final_code)

    print(f"\n✅ Playwright spec generated successfully:\n{OUTPUT_SPEC}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    print("📖 Reading Excel test cases...")

    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"❌ Excel file not found: {EXCEL_FILE}")

    test_cases = read_excel(EXCEL_FILE)

    if not test_cases:
        raise RuntimeError("❌ No test cases found in Excel")

    print(f"✅ Loaded {len(test_cases)} test cases")
    generate_spec(test_cases)
