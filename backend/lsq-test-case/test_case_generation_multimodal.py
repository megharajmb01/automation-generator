import os
from google import genai
from PIL import Image
from typing import Dict, Any, List
import json
import pandas as pd
from pydantic import BaseModel, Field

class TestCase(BaseModel):
    """Defines the structure for a single test case."""
    test_case_no: str = Field(description="Unique identifier for the test case (e.g., AC-TC-001).")
    test_description: str = Field(description="A clear, concise summary of the test's objective.")
    User_Role: str = Field(description="The user persona performing the test (e.g., Customer, Underwriter).")
    tab_flows: str = Field(description="The sequence of tabs/screens involved in this test (e.g., Tab A -> Tab B).")
    test_steps: List[str] = Field(description="The detailed, numbered sequence of actions a tester must perform.")
    expected_result: str = Field(description="The precise, observable outcome that validates the test.")

    class Config:
        extra = 'ignore'

class TestCasesOutput(BaseModel):
    """Defines the top-level structure (to match the model's preferred wrapper)."""
    test_cases: List[TestCase] = Field(description="The generated list of test cases.")

def clean_json_string(json_string: str) -> str:
    """Removes common Markdown code block wrappers from a JSON string."""
    cleaned = json_string.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()

def save_to_excel(positive_tcs: List[Dict[str, Any]], negative_tcs: List[Dict[str, Any]],
                  filename: str = "TestCases.xlsx"):
    """
    Converts two lists of test cases into a pandas DataFrame and saves them
    to two separate sheets in an Excel file.
    """
    print(f"\n--- SAVING TO {filename} ---")

    if not positive_tcs and not negative_tcs:
        print("No test cases generated. Skipping Excel file creation.")
        return

    try:
        writer = pd.ExcelWriter(filename, engine='openpyxl')

        # Helper to process and save a sheet
        def process_and_save(tcs, sheet_name):
            if tcs:
                df = pd.DataFrame([tc.model_dump() for tc in tcs])
                # Convert the 'test_steps' list into a readable, numbered string for Excel
                df['test_steps'] = df['test_steps'].apply(
                    lambda steps: '\n'.join([f"{i + 1}. {step}" for i, step in enumerate(steps)]) if isinstance(steps, list) else steps
                )
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Sheet '{sheet_name}' saved with {len(tcs)} cases.")

        process_and_save(positive_tcs, 'Positive_Test_Cases')
        process_and_save(negative_tcs, 'Negative_Test_Cases')

        writer.close()
        print(f"SUCCESS: Test cases saved to '{filename}'")

    except Exception as e:
        print(f"ERROR: Failed to save Excel file. Ensure 'openpyxl' is installed. Details: {e}")

def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """
    Load PIL Image objects from a list of image paths.
    Handles nested lists and validates file existence.
    
    Args:
        image_paths: List of image file paths (can be nested lists)
        
    Returns:
        List of loaded PIL Image objects
    """
    loaded_images = []
    
    if not image_paths:
        print("⚠️  No image paths provided")
        return loaded_images
    
    # Flatten nested lists
    def flatten(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result
    
    flat_paths = flatten(image_paths)
    
    print(f"📸 Loading {len(flat_paths)} images...")
    
    for idx, path in enumerate(flat_paths):
        if not isinstance(path, str):
            print(f"   ✗ Invalid path type at index {idx}: {type(path)}")
            continue
            
        if not os.path.exists(path):
            print(f"   ✗ Image not found: {path}")
            continue
        
        try:
            img = Image.open(path)
            loaded_images.append(img)
            print(f"   ✓ Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"   ✗ Error loading {path}: {e}")
    
    print(f"✓ Successfully loaded {len(loaded_images)}/{len(flat_paths)} images")
    return loaded_images

def generate_multimodal_test_cases(
    image_paths: List[str],
    unstructured_text: str,
    prompt_type: str = "POSITIVE"
) -> List[TestCase]:
    """
    Creates system/user prompts and calls the Gemini API with images and text,
    using Pydantic for schema enforcement and validation.
    
    Args:
        image_paths: List of image file paths to analyze
        unstructured_text: Text content from PDF section
        prompt_type: "POSITIVE" or "NEGATIVE" test cases
        
    Returns:
        List of TestCase objects or None on error
    """
    try:
        client = genai.Client(api_key=os.getenv("API_KEY"))
        model_name = "gemini-2.5-pro"

        # 1. Validate inputs and estimate complexity
        text_length = len(unstructured_text.strip())
        
        if not unstructured_text or text_length < 50:
            print(f"⚠️  WARNING: Text content is too short ({text_length} chars)")
            print("   Cannot generate meaningful test cases without sufficient context")
            return []

        # 2. Load images
        print(f"📂 Processing image paths: {image_paths}")
        loaded_images = load_images_from_paths(image_paths)
        
        if not loaded_images:
            print("⚠️  WARNING: No images loaded")

        # 3. Estimate appropriate number of test cases based on content
        # Calculate complexity score
        num_images = len(loaded_images)
        
        # Count key indicators of complexity
        num_fields = unstructured_text.lower().count('field') + unstructured_text.lower().count('input')
        num_validations = unstructured_text.lower().count('validat') + unstructured_text.lower().count('check')
        num_workflows = unstructured_text.lower().count('step') + unstructured_text.lower().count('flow')
        
        # Estimate test case count
        if text_length < 500 and num_images == 0:
            target_count = "5-8"
            complexity = "LOW"
        elif text_length < 1000 and num_images <= 1:
            target_count = "8-12"
            complexity = "MEDIUM-LOW"
        elif text_length < 2000 and num_images <= 2:
            target_count = "12-20"
            complexity = "MEDIUM"
        elif text_length < 5000 and num_images <= 4:
            target_count = "20-30"
            complexity = "MEDIUM-HIGH"
        else:
            target_count = "30-40"
            complexity = "HIGH"
        
        print(f"📊 Content Analysis:")
        print(f"📊 Content :", unstructured_text)
        print(f"   Text length: {text_length} chars")
        print(f"   Images: {num_images}")
        print(f"   Complexity: {complexity}")
        print(f"   Target test cases: {target_count}")

        # 4. Define the System Prompt with dynamic count
        system_prompt = (
            f"You are a Senior QA Engineer. Your task is to generate a comprehensive set of **{prompt_type}** test cases. "
            f"You must synthesize requirements from the provided text and any process flowchart images. "
            f"**CRITICAL RULES:**\n"
            f"1. Generate test cases ONLY based on the content provided in the text and images\n"
            f"2. Do NOT invent features, screens, or functionality not mentioned in the requirements\n"
            f"3. If images show process flows: Green tabs are accessible, red tabs are inaccessible\n"
            f"4. Test Case Numbers must be unique and sequential (e.g., AC-TC-001, AC-TC-002)\n"
            f"5. **Generate approximately {target_count} test cases** - scale based on available requirements\n"
            f"6. If content is limited, focus on quality over quantity - better to have 5 excellent test cases than 20 generic ones\n"
            f"7. The `test_steps` list MUST be detailed and specific to the actual requirements\n"
            f"8. Focus equally on field validations AND system workflows shown in diagrams\n"
            f"**Strictly adhere to the Pydantic schema provided.**"
        )

        # 4. Build user prompt WITHOUT sample test case
        user_prompt_text = f"""
### Requirements Document Content:
---
{unstructured_text}
---

### Instructions:
1. **Analyze the text** for all functional requirements, fields, validations, and workflows
2. **Analyze the images** (if provided) for process flows, screen sequences, and state transitions
3. **Generate test cases** that cover:
   - Field validations (data type, format, mandatory/optional, length limits)
   - Workflow steps (following the sequence shown in flowcharts)
   - State transitions (tab/screen accessibility based on process stage)
   - User role permissions
   - Integration points (APIs, external systems)

### Test Case Structure:
- **test_case_no**: Sequential ID (AC-TC-001, AC-TC-002, ...)
- **test_description**: Clear objective of the test
- **User_Role**: The persona performing the test
- **tab_flows**: Tab/screen navigation sequence
- **test_steps**: Detailed, numbered steps (be specific about data entry, clicks, validations)
- **expected_result**: Observable outcome that validates success

### Test Type: {prompt_type}
- For **POSITIVE** tests: Focus on successful paths, valid data, expected workflows
- For **NEGATIVE** tests: Focus on invalid data, boundary conditions, error scenarios

### Coverage Guidelines:
**Target: {target_count} test cases** (Content complexity: {complexity})

**IMPORTANT:** 
- If the requirements are limited, generate fewer high-quality test cases
- Don't pad with generic or repetitive tests
- Each test case should cover a distinct scenario from the requirements
- If you can only identify 3-5 meaningful test scenarios, that's perfectly acceptable
- Quality and specificity are more important than hitting a target number

**Generate test cases now based ONLY on the provided content.**
"""

        # 5. Build content list for API call
        content_parts = [user_prompt_text]
        content_parts.extend(loaded_images)
        # 6. Invoke the Gemini API
        print(f"🤖 Invoking Gemini API with model: {model_name}...")
        print(f"   Content parts: 1 text + {len(loaded_images)} images")
        print(f"   Expected test cases: {target_count}")

        response = client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_schema=TestCasesOutput,
                response_mime_type="application/json"
            )
        )

        json_text = response.text
        if not json_text:
            raise ValueError("Gemini returned an empty response text.")

        # 7. Clean and Validate with Pydantic
        cleaned_json_text = clean_json_string(json_text)
        parsed_output: TestCasesOutput = TestCasesOutput.model_validate_json(cleaned_json_text)
        
        actual_count = len(parsed_output.test_cases)
        print(f"✅ Generated {actual_count} test cases (target was {target_count})")
        
        # Warn if significantly more than expected
        max_expected = int(target_count.split('-')[1]) if '-' in target_count else 40
        if actual_count > max_expected * 1.5:
            print(f"⚠️  Warning: Generated {actual_count} test cases, expected ~{target_count}")
            print(f"   This might indicate the AI is padding with generic tests")
        
        return parsed_output.test_cases

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_generations_2(section_title: str, image_paths: List[str], section_text: str):
    """
    Wrapper function to generate both positive and negative test cases.
    
    Args:
        section_title: Title of the section being processed
        image_paths: List of image file paths
        section_text: Text content from the section
    """
    print(f"\n{'='*60}")
    print(f"GENERATING TEST CASES FOR: {section_title}")
    print(f"{'='*60}")
    print(f"📊 Text length: {len(section_text)} characters")
    print(f"📸 Image paths provided: {len(image_paths)}")
    

    # Generate positive test cases
    positive_tcs = generate_multimodal_test_cases(image_paths, section_text, "POSITIVE")
    if positive_tcs:
        print(f"✅ POSITIVE TEST CASES GENERATED ({len(positive_tcs)} cases).")
    else:
        print("⚠️  No positive test cases generated")

    # Generate negative test cases
    negative_tcs = generate_multimodal_test_cases(image_paths, section_text, "NEGATIVE")
    if negative_tcs:
        print(f"✅ NEGATIVE TEST CASES GENERATED ({len(negative_tcs)} cases).")
    else:
        print("⚠️  No negative test cases generated")

    # Save to Excel only if we have test cases
    if positive_tcs or negative_tcs:
        save_to_excel(positive_tcs, negative_tcs, "Generated_Test_Cases_multi_Pydantic.xlsx")
    else:
        print("\n⚠️  No test cases generated - skipping Excel file creation")

def run_generation(image_analysis_results: List[str], section_text: str, section_title: str):
    """
    Legacy wrapper function for compatibility.
    Note: image_analysis_results should be file paths, not analysis text.
    """
    run_generations_2(section_title, image_analysis_results, section_text)