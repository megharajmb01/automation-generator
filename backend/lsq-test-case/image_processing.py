import os
import requests
import json
import base64
import time
from typing import Dict, Any, List

API_KEY = os.getenv("API_KEY") # Replace with API KEY
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"


def encode_image_to_base64(image_path: str, mime_type: str = "image/png") -> str:
    """
    Encodes an image file into a Base64 string for API submission.
    
    Args:
        image_path: The file path to the image (e.g., 'flowchart.png').
        mime_type: The MIME type of the image (e.g., 'image/png', 'image/jpeg').
        
    Returns:
        The Base64 encoded string of the image data.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Note: The Base64 string does not include the mime type prefix here, 
            # as the payload structure handles that separately.
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
        return ""


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "metadata": {
            "type": "OBJECT",
            "description": "General descriptive information about the image/diagram.",
            "properties": {
                "diagram_type": {"type": "STRING", "description": "The type of diagram (e.g., Flowchart, UI Sketch, Sequence Diagram)."},
                "main_subject": {"type": "STRING", "description": "The primary topic or system represented (e.g., User Onboarding, E-commerce Checkout)."},
                "complexity_level": {"type": "STRING", "description": "Subjective complexity rating (e.g., Simple, Moderate, High)."}
            },
            "propertyOrdering": ["diagram_type", "main_subject", "complexity_level"]
        },
        "user_roles": {
            "type": "ARRAY",
            "description": "A list of users or actors identified in the process flow.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role_name": {"type": "STRING", "description": "The name of the user or actor (e.g., Guest User, Admin, Database)."},
                    "responsibility": {"type": "STRING", "description": "The main action or responsibility of this role in the diagram."}
                },
                "propertyOrdering": ["role_name", "responsibility"]
            }
        },
        "flow_steps": {
            "type": "ARRAY",
            "description": "A sequential breakdown of the process flow shown in the diagram.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "step_number": {"type": "NUMBER", "description": "The sequential number of the step."},
                    "action": {"type": "STRING", "description": "The action performed at this step."},
                    "condition": {"type": "STRING", "description": "The condition required or checked at this step (if applicable, otherwise empty)."},
                    "next_step_on_success": {"type": "STRING", "description": "What happens after this step succeeds."}
                },
                "propertyOrdering": ["step_number", "action", "condition", "next_step_on_success"]
            }
        }
    },
    "propertyOrdering": ["metadata", "user_roles", "flow_steps"]
}


def analyze_image_with_schema(base64_image_data: str, mime_type: str) -> Dict[str, Any]:
    """
    Calls the Gemini API to analyze an image and return structured JSON.
    """
    
    # 3a. Define the Prompts
    system_prompt = (
        "You are an expert diagram and flowchart analysis engine. "
        "Your task is to analyze the provided image, which depicts a process flow, "
        "and accurately extract the flow steps, user roles, and descriptive metadata "
        "into a structured JSON object according to the provided schema. "
        "Be detailed and precise in your extraction."
        "For clarity, a colour code is used,green tabs are visible and accessible to the user, while red tabs remain inaccessible until a laterstage. This way, we can clearly see how the system unfolds stage by stage and how the application progresses across functions."
    )
    user_query = "Analyze this diagram: Identify the complete process flow, all involved user roles, and provide a general summary of the diagram's purpose."

    # 3b. Construct the API Payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA
        },
    }

    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided (though handled by canvas in this environment)
    if API_KEY:
        API_URL_WITH_KEY = f"{API_URL}?key={API_KEY}"
    else:
        API_URL_WITH_KEY = API_URL
        
    # 3c. Execute API Call with Backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Calling Gemini API...")
            response = requests.post(API_URL_WITH_KEY, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # Extract JSON text from the response structure
            result = response.json()
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Parse the JSON text into a Python object
            parsed_data = json.loads(json_text)
            return parsed_data
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP Error: {e}")
                print(f"Response Status: {response.status_code}")
                print(f"Response Body: {response.text}")
                return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}
            
    return {} # Return empty on failure


def run_analysis(image_path: str, mime_type: str = "image/png"):
    """
    Runs the full analysis pipeline for a given image path.
    """

    print("--- STARTING IMAGE ANALYSIS ---")
    base64_data = encode_image_to_base64(image_path)

    if not base64_data:
        print("Invalid or missing Base64 data.")
        return {}

    result = analyze_image_with_schema(base64_data, mime_type)

    if result:
        print("\n--- ANALYSIS COMPLETE ---")
        print(result)
        print(json.dumps(result, indent=4))
        return result
    else:
        print("Analysis failed.")
        return {}


if __name__ == "__main__":
    IMAGE_FILE_PATH = "extracted_images_by_section/Application_Creation:/page18_img3.png"
    run_analysis(IMAGE_FILE_PATH)





   
#     IMAGE_FILE_PATH = "extracted_images_method/page3_img3.png"
#     IMAGE_MIME_TYPE = "image/png"
    
#     # --- DUMMY DATA FOR DEMONSTRATION ---
#     # Since we cannot access a local file system here, we must use a placeholder.
#     # IN A REAL ENVIRONMENT, you would use the 'encode_image_to_base64' function above.
    
#     print("--- STARTING IMAGE ANALYSIS DEMO ---")
#     print(f"Note: This script uses a placeholder for the Base64 image data.")
#     print("      In a real application, you would encode your image file.")
    
#     BASE64_IMAGE_PLACEHOLDER = "PLACEHOLDER_BASE64_DATA" 

#     BASE64_IMAGE_PLACEHOLDER = encode_image_to_base64(IMAGE_FILE_PATH)

    
#     analysis_result = analyze_image_with_schema(BASE64_IMAGE_PLACEHOLDER, IMAGE_MIME_TYPE)
    
#     if analysis_result:
#         print("\n--- LLM ANALYSIS COMPLETE ---")
#         print("\n--- RAW JSON OUTPUT ---")
#         print(json.dumps(analysis_result, indent=4))
        
#         # Nicely formatted output
#         print("\n--- STRUCTURED DATA EXTRACTION ---")
        
#         metadata = analysis_result.get('metadata', {})
#         print(f"\n[METADATA]")
#         print(f"  Diagram Type: {metadata.get('diagram_type')}")
#         print(f"  Subject:      {metadata.get('main_subject')}")
#         print(f"  Complexity:   {metadata.get('complexity_level')}")
        
#         users = analysis_result.get('user_roles', [])
#         print(f"\n[IDENTIFIED USERS/ACTORS]")
#         for user in users:
#             print(f"  - {user.get('role_name')}: {user.get('responsibility')}")
            
#         flow = analysis_result.get('flow_steps', [])
#         print(f"\n[PROCESS FLOW STEPS]")
#         for step in flow:
#             print(f"  Step {int(step.get('step_number', 0))}: {step.get('action')}")
#             if step.get('condition'):
#                 print(f"    - Condition: {step.get('condition')}")
#             print(f"    - Success leads to: {step.get('next_step_on_success')}")
            
#     else:
#         print("\nAnalysis failed. Check your API key, image path, and network connection.")












































# # VISION_EXTRACTION_PROMPT = """
# # You are a senior business analyst specializing in understanding BRD diagrams,
# # system architecture drawings, UI screens, and flow illustrations.

# # From this image, extract ALL possible information. Infer hidden insights when needed.

# # Return JSON with the following structure:

# # {
# #   "title": "",
# #   "image_summary": "",
# #   "actors": [],
# #   "systems": [],
# #   "ui_components": [],
# #   "flows": [
# #     { "from": "", "to": "", "description": "" }
# #   ],
# #   "external_integrations": [],
# #   "stages": [],
# #   "roles_and_permissions": [],
# #   "actions": [],
# #   "validations": [],
# #   "data_entities": [],
# #   "risks_or_failures": [],
# #   "business_rules": [],
# #   "unknown_or_uncertain_items": []
# # }

# # Rules:
# # - Use inference if labels are small
# # - Include EVERY arrow direction
# # - Include every user role represented visually
# # - Translate icons into meaning
# # - Use descriptive names when label missing
# # - DO NOT hallucinate outside the image context
# # - Prefer short array items; no paragraphs
# # """



# # from openai import OpenAI
# # import base64
# # from PIL import Image
# # import io
# # import json

# # client = OpenAI()

# # def encode_image(path):
# #     with open(path, "rb") as f:
# #         return base64.b64encode(f.read()).decode("utf-8")


# # def analyze_image(image_path):
# #     img_b64 = encode_image(image_path)

# #     response = client.chat.completions.create(
# #         model="gpt-5-vision",
# #         messages=[
# #             {
# #                 "role": "user",
# #                 "content": [
# #                     {"type": "text", "text": VISION_EXTRACTION_PROMPT},
# #                     {
# #                         "type": "image_url",
# #                         "image_url": f"data:image/png;base64,{img_b64}"
# #                     }
# #                 ]
# #             }
# #         ],
# #         temperature=0
# #     )

# #     try:
# #         return json.loads(response.choices[0].message.content)
# #     except:
# #         return response.choices[0].message.content


# # # Run analysis
# # result = analyze_image("brd_system_architecture.png")
# # print(json.dumps(result, indent=2))