# ============================================================================
# CRITICAL: Do this FIRST, before all other imports
# ============================================================================
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CHROMA_TELEMETRY'] = 'false'

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
print("CWD =", os.getcwd())
import uuid
import json
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb import PersistentClient
import hashlib
import requests
import urllib.request
from dotenv import load_dotenv
import re
import base64
import time
import tempfile
import pandas as pd
from google import genai
from google.genai import types

import section_subsection_processing as ssp
from image_processing import run_analysis
from test_case_generation_multimodal import run_generations_2, run_generation

load_dotenv()

FIGMA_API_BASE = os.getenv("FIGMA_API_BASE", "https://api.figma.com/v1")
FIGMA_TOKEN = os.getenv("FIGMA_TOKEN", "")

app = FastAPI(
    title="PDF Test Case Generation API",
    description="API for extracting sections from PDFs and generating test cases with persistent storage",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
processing_jobs = {}
chroma_client = PersistentClient(path="chroma_db")

# Persistent storage files
PROCESSING_METADATA_FILE = "pdf_processing_metadata.json"

figma_jobs = {}
FIGMA_MODE = os.getenv("FIGMA_MODE", "mock").lower()
LOCAL_JSON_DIR = "./figma_mocks"

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PDFUploadResponse(BaseModel):
    pdf_id: str
    message: str
    pdf_path: str
    is_reprocessed: bool

class SectionInfo(BaseModel):
    number: int
    level: int
    title: str
    parent_section_title: Optional[str]
    indexed_pages: str
    image_count: int
    section_number : str

class SectionsListResponse(BaseModel):
    pdf_id: str
    total_sections: int
    sections: List[SectionInfo]

class ProcessSectionRequest(BaseModel):
    pdf_id: str
    section_number: int

class ProcessSectionByTitleRequest(BaseModel):
    pdf_id: str
    section_title: str

class ProcessAllSubsectionsRequest(BaseModel):
    pdf_id: str
    section_title: str

class JobStatus(BaseModel):
    pdf_id: str
    status: str
    message: str
    sections_count: Optional[int] = None
    current_step: Optional[str] = None
    pdf_file: Optional[str] = None
    processed_at: Optional[str] = None

class ProcessingResult(BaseModel):
    pdf_id: str
    section_title: str
    status: str
    text_chunks_count: int
    image_count: int
    output_file: Optional[str] = None
    error: Optional[str] = None

class ProcessedPDFInfo(BaseModel):
    pdf_id: str
    pdf_file: str
    pdf_path: str
    processed_at: str
    sections_count: int
    total_images: int
    status: str

class PDFListResponse(BaseModel):
    total_pdfs: int
    pdfs: List[ProcessedPDFInfo]

class FigmaProjectRequest(BaseModel):
    figma_file_id: str
    figma_token: str

class FigmaFlowInfo(BaseModel):
    flow_id: str
    from_node: str
    to_node: str
    from_screen: str
    to_screen: str
    trigger: str
    action: str
    transition: str

class FigmaContextResponse(BaseModel):
    pdf_id: str
    figma_file_id: str
    flows: List[FigmaFlowInfo]
    images_count: int
    flows_summary: str
    status: str

class ProcessSectionWithFigmaRequest(BaseModel):
    pdf_id: str
    section_number: int
    figma_file_id: str
    figma_token: str


# ============================================================================
# FIGMA REQUEST/RESPONSE MODELS
# ============================================================================

class FigmaExtractSectionsRequest(BaseModel):
    file_key: str
    use_local: bool = True
    figma_token: Optional[str] = None

class FigmaGenerateTestCasesRequest(BaseModel):
    file_key: str
    section_id: str
    section_name: str
    google_api_key: str
    use_local: bool = True
    figma_token: Optional[str] = None

class FigmaJobStatus(BaseModel):
    job_id: str
    file_key: str
    status: str
    sections_count: Optional[int] = None
    current_section: Optional[str] = None
    output_file: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

class FigmaSectionInfo(BaseModel):
    section_id: str
    section_name: str
    type: str

class FigmaSectionsListResponse(BaseModel):
    job_id: str
    file_key: str
    total_sections: int
    sections: List[FigmaSectionInfo]


# ============================================================================
# FALLBACK SECTION EXTRACTION (for PDFs without TOC)
# ============================================================================

def extract_sections_without_toc(pdf_path: str, collection):
    """
    Extract sections from PDF without Table of Contents.
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"📄 PDF has {total_pages} pages (no TOC found - using fallback)")
        
        extracted_sections = []
        pages_per_section = max(1, total_pages // 5)
        
        section_number = 1
        for start_page in range(0, total_pages, pages_per_section):
            end_page = min(start_page + pages_per_section, total_pages)
            
            # Extract text from these pages
            section_text = ""
            for page_idx in range(start_page, end_page):
                page = reader.pages[page_idx]
                section_text += page.extract_text() + "\n"
            
            section_title = f"Section {section_number} (Pages {start_page + 1}-{end_page})"
            
            # Split into chunks for ChromaDB
            chunk_size = 1000
            chunks = [section_text[i:i + chunk_size] for i in range(0, len(section_text), chunk_size)]
            
            # FIX: Ensure metadata values are strings or numbers, never None
            metadata = {
                'section_title': section_title,
                'section_number': str(section_number),
                'level': '1',  # Convert to string
                'page_range': f"{start_page + 1}-{end_page}",
                'image_count': '0',  # Convert to string
                'parent_section_title': 'None',  # Convert None to string 'None'
            }
            
            # Add to ChromaDB
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():
                    doc_id = f"section_{section_number}_chunk_{chunk_idx}"
                    try:
                        collection.add(
                            ids=[doc_id],
                            documents=[chunk],
                            metadatas=[metadata]
                        )
                    except Exception as e:
                        print(f"⚠️  Warning adding chunk {chunk_idx}: {e}")
            
            extracted_sections.append({
                'title': section_title,
                'level': 1,
                'indexed_pages': f"{start_page + 1}-{end_page}",
                'image_count': 0,
                'section_number': str(section_number),
                'parent_section_title': None
            })
            
            print(f"✓ Added {section_title}")
            section_number += 1
        
        return extracted_sections
        
    except Exception as e:
        print(f"✗ Error in fallback extraction: {e}")
        raise Exception(f"Failed to extract sections: {str(e)}")
    
# ============================================================================
# FIGMA HELPER FUNCTIONS (NEW)
# ============================================================================

def sanitize_figma_name(name: str) -> str:
    """Sanitize filenames for Figma"""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).lower().strip("_")

def load_figma_local_json(file_key: str) -> dict:
    """Load Figma JSON from local mock"""
    path = Path(LOCAL_JSON_DIR) / f"{file_key}.json"
    print(f"[LOCAL JSON] Loading: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Local JSON not found: {path}")
    return json.loads(path.read_text())

def get_figma_file(file_key: str, token: str | None, use_local: bool) -> dict:
    """Fetch Figma file (local mock or API)"""
    if use_local:
        return load_figma_local_json(file_key)

    if not token:
        raise ValueError("Figma token required for API requests")

    print(f"[FIGMA API] Fetching file: {file_key}")
    r = requests.get(
        f"{FIGMA_API_BASE}/files/{file_key}",
        headers={"X-Figma-Token": token}
    )
    r.raise_for_status()
    return r.json()

def extract_figma_sections(node: dict, out: dict):
    """Recursively extract frame/section nodes from Figma document"""
    if node.get("type") in ("FRAME", "GROUP", "SECTION") and node.get("name"):
        out[node["id"]] = {"name": node["name"], "type": node["type"]}
    for child in node.get("children", []):
        extract_figma_sections(child, out)

def load_or_recreate_figma_job(job_id: str) -> Optional[Dict]:
    """Load Figma job from memory or recreate from storage"""
    if job_id in figma_jobs:
        return figma_jobs[job_id]
    
    job_dir = Path(f"figma_jobs/{job_id}")
    if not job_dir.exists():
        return None
    
    metadata_file = job_dir / "job_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            job = json.load(f)
        figma_jobs[job_id] = job
        return job
    
    return None

def save_figma_job_metadata(job_id: str, job: Dict):
    """Save Figma job metadata to disk"""
    job_dir = Path(f"figma_jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = job_dir / "job_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(job, f, indent=2)

def create_figma_job_directory(job_id: str) -> str:
    """Create a directory for Figma job-specific files"""
    job_dir = Path(f"figma_jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    return str(job_dir)

# ============================================================================
# FIGMA TEST CASE GENERATION FUNCTIONS
# ============================================================================

FIGMA_SYSTEM_PROMPT = """
You are a Senior QA Engineer specializing in workflow-driven enterprise platforms.
Interpret workflow diagrams as functional BRDs.

Consider:
- actors (user/system/vendor/service/queue/bot)
- decisions, conditions, branches
- modes and variants
- dependencies and handoffs
- validation and sequencing
- state transitions and terminal outcomes

You will generate structured functional test cases.
Output JSON only. No comments, no markdown, no surrounding text.

Output format:
{
  "test_cases": [
    {
      "test_case_no": "...",
      "Module": "..."
      "test_description": "...",
      "User_Role": "...",
      "tab_flows": "...",
      "test_steps": ["...", "..."],
      "expected_result": "..."
    }
  ]
}
"""

def build_figma_user_prompt(section_name: str, prompt_type: str) -> str:
    """Build the user prompt for Figma test case generation"""
    return f"""
Section: {section_name}

Task:
Generate {prompt_type} functional test cases from the workflow diagram image.

Interpretation Phase (internal):
- identify actors
- extract decisions, conditions, branches
- infer possible modes/variants
- identify preconditions and sequencing rules
- detect dependencies/handoffs across actors
- determine success vs failure terminal states

Test Generation Rules:
- test_case_no sequential (e.g., AC-TC-001, ...)
- User_Role reflects actor (human/system/vendor/service/etc)
- tab_flows represents sequential navigation
- test_steps granular (inputs, actions, handoffs)
- expected_result single observable outcome

Case Volume:
Generate 20–40 cases depending on complexity.
"""

async def generate_figma_cases(
    image_b64: str,
    section_name: str,
    prompt_type: str,
    api_key: str
) -> tuple[list[dict], int]:
    """Generate test cases using Google Gemini from Figma image"""
    print(f"[LLM] Calling gemini-2.5-pro ({prompt_type})")

    client = genai.Client(api_key=os.getenv("API_KEY"))
    img_bytes = base64.b64decode(image_b64)

    start = time.time()

    try:
        user_prompt = build_figma_user_prompt(section_name, prompt_type)
        full_prompt = FIGMA_SYSTEM_PROMPT + "\n\n" + user_prompt
        
        # FIX: Create a single Content object with parts array
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=types.Content(
                role="user",
                parts=[
                    types.Part(text=full_prompt),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=img_bytes
                        )
                    )
                ]
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        ms = int((time.time() - start) * 1000)
        txt = response.text.strip()

        # Clean up markdown formatting if present
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()

        try:
            data = json.loads(txt)
            cases = data.get("test_cases", [])
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON from LLM: {e}")
            print(f"Response text: {txt[:500]}")
            raise HTTPException(500, f"Invalid JSON from LLM: {str(e)}")

        print(f"[LLM] Completed: {len(cases)} cases in {ms}ms")
        return cases, ms
        
    except Exception as e:
        print(f"ERROR in generate_figma_cases: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_figma_excel(pos: list[dict], neg: list[dict], job_dir: str, section_name: str) -> str:
    """Save Figma test cases to Excel file and return filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FigmaTestCases_{sanitize_figma_name(section_name)}_{timestamp}.xlsx"
    filepath = Path(job_dir) / filename

    print(f"[EXCEL] Saving: {filepath}")
    writer = pd.ExcelWriter(filepath, engine="openpyxl")

    def write_sheet(arr: list[dict], sheet_name: str):
        if not arr:
            print(f"[EXCEL] Skipping empty sheet: {sheet_name}")
            return

        df = pd.DataFrame(arr)

        if "test_steps" in df.columns:
            df["test_steps"] = df["test_steps"].apply(
                lambda steps: "\n".join(
                    [f"{i+1}. {s}" for i, s in enumerate(steps)]
                ) if isinstance(steps, list) else steps
            )

        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[EXCEL] Wrote {len(df)} rows to sheet: {sheet_name}")

    write_sheet(pos, "Positive_Test_Cases")
    write_sheet(neg, "Negative_Test_Cases")
    writer.close()
    
    return filename

# =============================================================================
# PERSISTENCE HELPER FUNCTIONS
# =============================================================================

def get_pdf_file_hash(pdf_path: str) -> str:
    """Generate hash of PDF file for validation"""
    hash_md5 = hashlib.md5()
    try:
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Error hashing file: {e}")
        return ""
    return hash_md5.hexdigest()

def load_processing_metadata() -> Dict:
    """Load metadata about processed PDFs"""
    if os.path.exists(PROCESSING_METADATA_FILE):
        try:
            with open(PROCESSING_METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processing metadata: {e}")
            return {}
    return {}

def save_processing_metadata(metadata: Dict):
    """Save metadata about processed PDFs"""
    try:
        with open(PROCESSING_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving processing metadata: {e}")

def check_if_pdf_processed(pdf_path: str, pdf_filename: str) -> tuple:
    """Check if PDF has already been processed"""
    metadata = load_processing_metadata()
    pdf_hash = get_pdf_file_hash(pdf_path)
    
    for pdf_id, info in metadata.items():
        if info.get('pdf_hash') == pdf_hash and info.get('pdf_file') == pdf_filename:
            return pdf_id, info
    
    return None, None

def register_pdf_processing(pdf_path: str, pdf_filename: str, pdf_id: str, 
                           sections_count: int, image_count: int) -> str:
    """Register a PDF as processed with metadata"""
    metadata = load_processing_metadata()
    
    metadata[pdf_id] = {
        'pdf_file': pdf_filename,
        'pdf_path': pdf_path,
        'pdf_hash': get_pdf_file_hash(pdf_path),
        'pdf_id': pdf_id,
        'processed_at': datetime.now().isoformat(),
        'sections_count': sections_count,
        'total_images': image_count,
        'status': 'completed'
    }
    
    save_processing_metadata(metadata)
    return pdf_id

def get_pdf_by_id(pdf_id: str) -> Optional[Dict]:
    """Retrieve PDF processing information by pdf_id"""
    metadata = load_processing_metadata()
    return metadata.get(pdf_id)

def list_all_processed_pdfs() -> Dict:
    """List all processed PDFs with their IDs"""
    metadata = load_processing_metadata()
    return metadata

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_job_directory(pdf_id: str) -> str:
    """Create a directory for pdf-specific files"""
    job_dir = Path(f"jobs/{pdf_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    return str(job_dir)

def get_section_map(extracted_sections: List[Dict]) -> Dict[int, Dict]:
    """Create a mapping of section numbers to section data"""
    section_map = {}
    for idx, section in enumerate(extracted_sections, 1):
        section_map[idx] = {
            'title': section['title'],
            'level': section['level'],
            'indexed_pages': section['indexed_pages'],
            'image_count': section['image_count'],
            'parent_section_title': section.get('parent_section_title', ''),
            'section_number': section['section_number']
        }
    return section_map

def load_or_recreate_job(pdf_id: str) -> Optional[Dict]:
    """
    Load a job from memory or recreate it from persistent storage.
    Returns the job dict or None if not found.
    """
    # Check if job is in memory
    if pdf_id in processing_jobs:
        return processing_jobs[pdf_id]
    
    # Try to load from persistent storage
    pdf_info = get_pdf_by_id(pdf_id)
    if not pdf_info:
        return None
    
    try:
        collection_name = f"pdf_{pdf_id}"
        collection = chroma_client.get_collection(name=collection_name)
        
        results = collection.get(limit=10000, include=['metadatas'])
        if not results['metadatas']:
            return None
        
        # Build section map from metadata
        sections_dict = {}
        for metadata in results['metadatas']:
            section_title = metadata.get('section_title', '')
            if section_title not in sections_dict:
                sections_dict[section_title] = {
                    'title': section_title,
                    'level': int(metadata.get('level', 0)),
                    'indexed_pages': metadata.get('page_range', ''),
                    'image_count': int(metadata.get('image_count', 0)),
                    'parent_section_title': metadata.get('parent_section_title', ''),
                    'section_number': metadata.get('section_number', '')
                }
        
        job_dir = f"jobs/{pdf_id}"
        job = {
            "status": "sections_extracted",
            "pdf_path": pdf_info['pdf_path'],
            "pdf_filename": pdf_info['pdf_file'],
            "job_dir": job_dir,
            "sections": list(sections_dict.values()),
            "section_map": {i+1: v for i, v in enumerate(sections_dict.values())},
            "collection": collection,
            "collection_name": collection_name
        }
        processing_jobs[pdf_id] = job
        return job
        
    except Exception as e:
        print(f"Error loading job {pdf_id}: {str(e)}")
        return None


# =============================================================================
# NEW FIGMA FLOW EXTRACTOR CLASS
# =============================================================================

def get_figma_mock_path(figma_file_id: str) -> str:
    mock_dir = os.getenv("FIGMA_MOCK_DIR", "figma_mocks")
    Path(mock_dir).mkdir(exist_ok=True)
    return os.path.join(mock_dir, f"{figma_file_id}.json")


class FigmaFlowExtractor:
    """Extract prototype flows from Figma JSON data."""
    
    def __init__(self):
        self.node_map = {}
        self.flows = []

    def traverse_node(self, node: Dict[str, Any], path: List[str] = None, parent_screen: str = None) -> None:
        """Recursively traverse nodes to find prototype interactions."""
        if path is None:
            path = []
        
        current_screen = parent_screen
        if node.get('type') in ['FRAME', 'COMPONENT', 'INSTANCE'] and len(path) <= 3:
            current_screen = node.get('name', 'Unnamed')
        
        if 'id' in node:
            self.node_map[node['id']] = {
                'id': node['id'],
                'name': node.get('name', 'Unnamed'),
                'type': node.get('type', 'UNKNOWN'),
                'parent_screen': current_screen
            }

        interactions = []
        if 'interactions' in node:
            interactions = node['interactions']
        elif 'reactions' in node:
            interactions = node['reactions']

        for idx, interaction in enumerate(interactions):
            trigger = interaction.get('trigger', {})
            action = interaction.get('action', {}) or (interaction.get('actions', [{}])[0] if interaction.get('actions') else {})
            
            if not isinstance(action, dict):
                action = {}
            
            transition_info = action.get('transition', {})
            if not isinstance(transition_info, dict):
                transition_info = {}
            
            flow = {
                'id': f"{node['id']}-{idx}",
                'from_node': node.get('name', node.get('id', 'Unnamed')),
                'from_node_id': node['id'],
                'from_node_type': node.get('type', 'UNKNOWN'),
                'from_screen': current_screen or 'Unknown Screen',
                'trigger': trigger.get('type') if isinstance(trigger, dict) else (trigger if isinstance(trigger, str) else 'UNKNOWN'),
                'action': action.get('type', 'UNKNOWN') if isinstance(action, dict) else (action if isinstance(action, str) else 'UNKNOWN'),
                'to_node_id': action.get('destinationId'),
                'transition': transition_info.get('type', 'NONE') if isinstance(transition_info, dict) else 'NONE',
                'duration': transition_info.get('duration'),
                'easing': transition_info.get('easing', {}).get('type') if isinstance(transition_info, dict) and isinstance(transition_info.get('easing'), dict) else None,
                'path': path + [node.get('name', node.get('id', 'Unnamed'))]
            }
            self.flows.append(flow)

        if 'children' in node and isinstance(node['children'], list):
            current_path = path + [node.get('name', node.get('id', 'Unnamed'))]
            for child in node['children']:
                self.traverse_node(child, current_path, current_screen)

    def extract_flows(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all prototype flows from Figma data."""
        self.flows = []
        self.node_map = {}

        if 'document' in data:
            self.traverse_node(data['document'])
        elif 'children' in data:
            for child in data['children']:
                self.traverse_node(child)
        else:
            self.traverse_node(data)

        return self.enrich_flows()

    def enrich_flows(self) -> List[Dict[str, Any]]:
        """Add destination node names and screens to flows."""
        enriched = []
        for flow in self.flows:
            flow_copy = flow.copy()
            if flow['to_node_id'] and flow['to_node_id'] in self.node_map:
                to_node_info = self.node_map[flow['to_node_id']]
                flow_copy['to_node'] = to_node_info['name']
                flow_copy['to_screen'] = to_node_info.get('parent_screen', 'Unknown Screen')
            else:
                flow_copy['to_node'] = 'Unknown'
                flow_copy['to_screen'] = 'Unknown Screen'
            enriched.append(flow_copy)
        return enriched


# =============================================================================
# NEW FIGMA HELPER FUNCTIONS
# =============================================================================

def download_figma_images(figma_file_id: str, figma_token: str, job_dir: str, limit: int = 10) -> List[str]:
    """Download images from Figma file and save locally, return absolute paths."""
    downloaded_images = []
    
    try:
        headers = {'X-Figma-Token': figma_token}
        
        # Get images metadata
        images_response = requests.get(
            f"{FIGMA_API_BASE}/files/{figma_file_id}/images",
            headers=headers,
            timeout=30
        )
        images_response.raise_for_status()
        images_data = images_response.json()
        
        if 'error' in images_data and images_data['error']:
            print(f"Figma API error: {images_data.get('status', 'Unknown')}")
            return []
        
        # Create figma_assets subdirectory with ABSOLUTE path
        job_dir_abs = os.path.abspath(job_dir)
        figma_assets_dir = os.path.join(job_dir_abs, "figma_assets")
        os.makedirs(figma_assets_dir, exist_ok=True)
        
        # Download images
        image_urls = images_data.get('meta', {}).get('images', {})
        
        for idx, (node_id, image_url) in enumerate(list(image_urls.items())[:limit]):
            try:
                filename = f"figma_screen_{idx}.png"
                # Use ABSOLUTE path
                filepath = os.path.join(figma_assets_dir, filename)
                
                urllib.request.urlretrieve(image_url, filepath)
                # Return absolute path
                downloaded_images.append(filepath)
                print(f"✓ Downloaded Figma image {idx + 1}: {filename}")
            except Exception as e:
                print(f"✗ Error downloading Figma image {idx}: {e}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Figma images: {e}")
    except Exception as e:
        print(f"Error in download_figma_images: {e}")
    
    return downloaded_images

def fetch_figma_flows(figma_file_id: str, figma_token: str) -> tuple:
    """
    Fetch Figma flows either from:
    - LIVE Figma API
    - LOCAL MOCK JSON (to avoid rate limits)
    """
    figma_mode = os.getenv("FIGMA_MODE", "live").lower()
    mock_path = get_figma_mock_path(figma_file_id)

    try:
        # ===============================
        # MOCK MODE
        # ===============================
        if figma_mode == "mock" and os.path.exists(mock_path):
            print(f"🧪 Using MOCK Figma JSON: {mock_path}")

            with open(mock_path, "r", encoding="utf-8") as f:
                figma_data = json.load(f)

        # ===============================
        # LIVE MODE
        # ===============================
        else:
            print(f"🌐 Fetching LIVE Figma API for {figma_file_id}")

            headers = {"X-Figma-Token": figma_token}
            response = requests.get(
                f"{FIGMA_API_BASE}/files/{figma_file_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            figma_data = response.json()

            # Save response for future mocking
            with open(mock_path, "w", encoding="utf-8") as f:
                json.dump(figma_data, f, indent=2)

            print(f"💾 Saved Figma mock JSON → {mock_path}")

        # ===============================
        # SHARED FLOW EXTRACTION
        # ===============================
        extractor = FigmaFlowExtractor()
        flows = extractor.extract_flows(figma_data)

        flows_summary = {
            "total_flows": len(flows),
            "screens": list(set(f["from_screen"] for f in flows)),
            "interactions": list(set(f["trigger"] for f in flows)),
            "sample_flows": [
                {
                    "from": f["from_screen"],
                    "to": f["to_screen"],
                    "trigger": f["trigger"],
                    "action": f["action"]
                }
                for f in flows[:20]
            ]
        }

        return flows, flows_summary

    except Exception as e:
        print(f"❌ Error fetching Figma flows: {e}")
        return [], {}

def create_figma_context_string(flows: List[Dict], flows_summary: Dict) -> str:
    """Create a context string from Figma flows for LLM."""
    context = f"""
UI PROTOTYPE INFORMATION (from Figma):
=====================================
Total Flows Detected: {flows_summary.get('total_flows', 0)}
Unique Screens: {', '.join(flows_summary.get('screens', []))}
User Interactions: {', '.join(flows_summary.get('interactions', []))}

NAVIGATION FLOWS:
"""
    
    for flow in flows_summary.get('sample_flows', [])[:10]:
        context += f"\n- {flow['from']} → {flow['to']} (trigger: {flow['trigger']}, action: {flow['action']})"
    
    return context


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "PDF Test Case Generation API with Persistent Storage is running",
        "version": "2.0.0"
    }

@app.post("/api/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing
    
    The PDF will be stored in: jobs/{pdf_id}/{filename}
    Automatically detects if PDF was previously processed
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique pdf_id (same as job_id)
    pdf_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = create_job_directory(pdf_id)
    
    # Save uploaded PDF
    pdf_path = os.path.join(job_dir, file.filename)
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"✅ PDF uploaded and saved to: {pdf_path}")
    
    # Check if PDF was previously processed
    existing_pdf_id, existing_info = check_if_pdf_processed(pdf_path, file.filename)
    
    is_reprocessed = False
    if existing_pdf_id:
        # Use existing pdf_id instead of generating new one
        pdf_id = existing_pdf_id
        is_reprocessed = True
        print(f"📦 PDF already processed with ID: {pdf_id}")
    
    # Initialize job tracking
    processing_jobs[pdf_id] = {
        "status": "uploaded",
        "pdf_path": pdf_path,
        "pdf_filename": file.filename,
        "job_dir": job_dir,
        "sections": None,
        "section_map": None,
        "collection": None,
        "is_reprocessed": is_reprocessed
    }
    
    message = f"PDF uploaded successfully. "
    if is_reprocessed:
        message += f"Previously processed with ID: {pdf_id}"
    else:
        message += f"Ready for extraction with ID: {pdf_id}"
    
    return PDFUploadResponse(
        pdf_id=pdf_id,
        message=message,
        pdf_path=pdf_path,
        is_reprocessed=is_reprocessed
    )

@app.post("/api/extract-sections/{pdf_id}", response_model=SectionsListResponse)
async def extract_sections(pdf_id: str):
    """
    Extract sections from PDF with automatic fallback.
    
    1. Try TOC-based extraction first
    2. If no TOC, use page-based fallback
    3. Return sections list for selection
    """
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    
    if job["status"] not in ["uploaded", "error"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is in '{job['status']}' state. Cannot re-extract."
        )
    
    job["status"] = "extracting"
    
    try:
        # Initialize ChromaDB collection
        collection_name = f"pdf_{pdf_id}"
        collection = chroma_client.get_or_create_collection(name=collection_name)
        job["collection"] = collection
        job["collection_name"] = collection_name
        
        print(f"📖 Extracting sections from {job['pdf_path']}")
        
        extracted_sections = None
        extraction_method = None
        
        # Method 1: Try TOC-based extraction
        try:
            print("  1️⃣  Attempting TOC-based extraction...")
            extracted_sections = ssp.get_toc_and_extract_data(
                pdf_path=job["pdf_path"],
                collection=collection,
                extract_images=True,
                filter_logos=True,
                min_image_size=0,
                exclude_header_footer=True
            )
            extraction_method = "TOC"
            print("  ✓ Success with TOC extraction")
        except Exception as toc_error:
            print(f"  ✗ TOC extraction failed: {str(toc_error)[:100]}")
        
        # Method 2: Fallback to page-based extraction
        if not extracted_sections:
            try:
                print("  2️⃣  Attempting page-based fallback extraction...")
                extracted_sections = extract_sections_without_toc(
                    pdf_path=job["pdf_path"],
                    collection=collection
                )
                extraction_method = "Page-based Fallback"
                print("  ✓ Success with fallback extraction")
            except Exception as fallback_error:
                print(f"  ✗ Fallback extraction failed: {str(fallback_error)}")
                raise
        
        if not extracted_sections:
            job["status"] = "error"
            raise HTTPException(
                status_code=500, 
                detail="Failed to extract sections from PDF"
            )
        
        # Store in job
        job["sections"] = extracted_sections
        job["section_map"] = get_section_map(extracted_sections)
        job["status"] = "sections_extracted"
        
        # Register processing
        total_images = sum(s.get('image_count', 0) for s in extracted_sections)
        register_pdf_processing(
            job["pdf_path"],
            job["pdf_filename"],
            pdf_id,
            len(extracted_sections),
            total_images
        )
        
        # Build response
        sections_info = []
        for num, section_data in job["section_map"].items():
            section = SectionInfo(
                number=num,
                level=section_data["level"],
                title=section_data["title"],
                parent_section_title=section_data.get("parent_section_title"),
                indexed_pages=section_data["indexed_pages"],
                image_count=section_data.get("image_count", 0),
                section_number=section_data["section_number"]
            )
            sections_info.append(section)
        
        print(f"✅ Extraction complete ({extraction_method})")
        print(f"   Extracted {len(extracted_sections)} sections")
        
        return SectionsListResponse(
            pdf_id=pdf_id,
            total_sections=len(extracted_sections),
            sections=sections_info
        )
        
    except HTTPException:
        job["status"] = "error"
        raise
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"❌ Extraction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to extract sections: {str(e)}"
        )

@app.post("/api/process-section", response_model=ProcessingResult)
async def process_section(request: ProcessSectionRequest):
    """
    Process a selected section by number to generate test cases.
    When a parent section is selected, it includes all subsection images.
    """
    pdf_id = request.pdf_id
    section_number = request.section_number
    
    job = load_or_recreate_job(pdf_id)
    if not job:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    if section_number not in job["section_map"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid section number. Must be between 1 and {len(job['section_map'])}"
        )
    
    job["status"] = "processing_section"
    job["current_section"] = section_number
    
    try:
        section_title = job["section_map"][section_number]['title']
        collection = job["collection"]
        section_info = job["section_map"][section_number]
        actual_section_number = section_info.get('section_number')  
        
        # DEBUG: Check what's in the collection
        ssp.debug_section_in_collection(collection, section_title)
        
        # Retrieve section data with images (includes subsections)
        section_data = ssp.get_section_data_with_images(
            collection=collection,
            section_title=section_title,
            section_number=actual_section_number,
            include_subsections=True  # Always include subsections
        )

        print(f"\n📊 Section data retrieved: {section_title}")
        print(f"   Text length: {len(section_data.get('full_text', ''))} characters")
        print(f"   Chunks: {len(section_data.get('text_chunks', []))}")
        print(f"   Direct images: {section_data.get('image_count', 0)}")
        print(f"   Subsections: {len(section_data.get('subsections', []))}")
        
        # Debug: Show subsection details
        for idx, subsection in enumerate(section_data.get('subsections', []), 1):
            print(f"      └─ Subsection {idx}: {subsection.get('section_title', 'Unknown')}")
            print(f"         Level: {subsection.get('level')}, Images: {len(subsection.get('images', []))}")
            if subsection.get('images'):
                for img in subsection.get('images', []):
                    print(f"            • {os.path.basename(img)}")
        
        # Collect ALL images - from main section AND all subsections
        all_image_paths = []
        
        # Add images from the main section
        if section_data.get('has_images'):
            main_images = section_data.get('images', [])
            all_image_paths.extend(main_images)
            print(f"   ├─ Main section images: {len(main_images)}")
        
        # Add images from all subsections
        for subsection in section_data.get('subsections', []):
            if subsection.get('has_images'):
                sub_images = subsection.get('images', [])
                all_image_paths.extend(sub_images)
                print(f"   ├─ Subsection '{subsection['section_title']}' images: {len(sub_images)}")
        
        print(f"   └─ Total images collected: {len(all_image_paths)}")
        
        # Process images - verify they exist and collect valid paths
        image_analysis_results = []
        missing_images = []
        
        for image_path in all_image_paths:
            if os.path.exists(image_path):
                image_analysis_results.append(image_path)
            else:
                missing_images.append(image_path)
                print(f"⚠️  Warning: Image not found: {image_path}")
        
        if missing_images:
            print(f"⚠️  {len(missing_images)} images not found on disk")
        
        print(f"✅ Valid images for processing: {len(image_analysis_results)}")
        
        # Collect full text including subsections
        full_text_parts = []
        
        # Add main section text (but only if it has substantial content)
        main_text = section_data.get('full_text', '').strip()
        if len(main_text) > 50:  # Only include if it has real content beyond titles
            full_text_parts.append(main_text)
        
        # Add subsection texts
        for subsection in section_data.get('subsections', []):
            subsection_text = subsection.get('full_text', '').strip()
            if subsection_text and len(subsection_text) > 20:  # Skip empty or title-only sections
                # Add section header for context
                subsection_title = subsection.get('section_title', 'Subsection')
                full_text_parts.append(f"\n\n## {subsection_title}\n\n{subsection_text}")
        
        combined_text = '\n'.join(full_text_parts)
        print(f"📝 Text combination:")
        print(f"   Main section text: {len(main_text)} chars")
        print(f"   Subsections added: {len(section_data.get('subsections', []))}")
        print(f"   Combined length: {len(combined_text)} characters")
        
        # Debug: Show first 500 chars of combined text
        if len(combined_text) > 0:
            preview = combined_text[:500].replace('\n', ' ')
            print(f"   Preview: {preview}...")
        else:
            print(f"   ⚠️  WARNING: No text content found!")
        
        # Generate test cases - CONVERT TO ABSOLUTE PATH
        job_dir_absolute = os.path.abspath(job["job_dir"])
        output_filename = f"TestCases_{pdf_id}_{section_title}.xlsx"
        output_path = os.path.join(job_dir_absolute, output_filename)
        
        original_dir = os.getcwd()
        
        try:
            # Change to job directory
            os.chdir(job_dir_absolute)
            
            print(f"📂 Working directory changed to: {os.getcwd()}")
            print(f"📂 Job directory (absolute): {job_dir_absolute}")
            print(f"📂 Output filename: {output_filename}")
            print(f"📂 Output path (absolute): {output_path}")
            
            # Call test case generation with ALL images
            print(f"🔄 Generating test cases...")
            print(f"   Section: {section_title}")
            print(f"   Images: {len(image_analysis_results)}")
            print(f"   Text length: {len(combined_text)}")
            
            run_generations_2(
                section_title, 
                image_analysis_results,  # All images from main + subsections
                combined_text  # Combined text from main + subsections
            )
            
            # The function saves as "Generated_Test_Cases_multi_Pydantic.xlsx" in current directory
            generated_file = os.path.join(job_dir_absolute, "Generated_Test_Cases_multi_Pydantic.xlsx")
            
            print(f"✓ Test case generation completed")
            print(f"  Generated file path: {generated_file}")
            
            # Check if file exists
            if os.path.exists(generated_file):
                print(f"✓ Found generated file")
                
                # Rename it to our target filename
                if generated_file != output_path:
                    import shutil
                    shutil.move(generated_file, output_path)
                    print(f"✓ Renamed to {output_filename}")
            else:
                # Search for xlsx files as fallback
                try:
                    files_in_dir = os.listdir(job_dir_absolute)
                    print(f"📋 Files in directory: {files_in_dir}")
                    
                    xlsx_files = [f for f in files_in_dir if f.endswith('.xlsx')]
                    
                    if xlsx_files:
                        print(f"⚠️  Standard filename not found, using: {xlsx_files[0]}")
                        found_file = os.path.join(job_dir_absolute, xlsx_files[0])
                        import shutil
                        shutil.move(found_file, output_path)
                        print(f"✓ Moved {xlsx_files[0]} to {output_filename}")
                    else:
                        raise FileNotFoundError(
                            f"No Excel files found in {job_dir_absolute}"
                        )
                except Exception as list_error:
                    print(f"✗ Error listing files: {list_error}")
                    raise FileNotFoundError(
                        f"Test case generation did not produce output file at {generated_file}"
                    )
                
        finally:
            # Always return to original directory
            os.chdir(original_dir)
            print(f"✓ Returned to original directory: {os.getcwd()}")
        
        # Final verification - use absolute path
        if not os.path.exists(output_path):
            print(f"❌ CRITICAL: Output file missing at {output_path}")
            raise FileNotFoundError(f"Output file not found at {output_path}")
        
        file_size = os.path.getsize(output_path)
        print(f"✅ SUCCESS: Test cases saved")
        print(f"   Filename: {output_filename}")
        print(f"   Path: {output_path}")
        print(f"   Size: {file_size} bytes")
        print(f"   Section: {section_title}")
        print(f"   Images processed: {len(image_analysis_results)}")
        print(f"   Subsections included: {len(section_data.get('subsections', []))}")
        
        job["status"] = "completed"
        job["output_file"] = output_filename
        
        return ProcessingResult(
            pdf_id=pdf_id,
            section_title=section_title,
            status="completed",
            text_chunks_count=len(section_data.get('text_chunks', [])),
            image_count=len(image_analysis_results),  # Total images including subsections
            output_file=output_filename
        )
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process section: {str(e)}"
        )

@app.post("/api/process-section-by-title", response_model=ProcessingResult)
async def process_section_by_title(request: ProcessSectionByTitleRequest):
    """
    Process a section by its title to generate test cases
    
    Endpoint that processes by exact section title instead of number
    """
    pdf_id = request.pdf_id
    section_title = request.section_title
    
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    
    if job["status"] != "sections_extracted":
        raise HTTPException(
            status_code=400,
            detail=f"Job is in '{job['status']}' state. Extract sections first."
        )
    
    # Find section by title
    section_exists = any(
        sec['title'] == section_title 
        for sec in job["section_map"].values()
    )
    
    if not section_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Section '{section_title}' not found"
        )
    
    job["status"] = "processing_section"
    
    try:
        collection = job["collection"]
        
        # Retrieve section data
        section_data = ssp.get_section_data_with_images(
            collection=collection,
            section_title=section_title,
            include_subsections=True
        )
        
        # Process images
        image_analysis_results = []
        if section_data['has_images']:
            for image_path in section_data['images']:
                if os.path.exists(image_path):
                    try:
                        analysis = run_analysis(image_path)
                        image_analysis_results.append(str(analysis))
                    except Exception as e:
                        print(f"Error analyzing image {image_path}: {e}")
        
        # Generate test cases
        output_filename = f"TestCases_{pdf_id}_{section_title.replace(' ', '_')}.xlsx"
        output_path = os.path.join(job["job_dir"], output_filename)
        
        original_dir = os.getcwd()
        os.chdir(job["job_dir"])
        
        try:
            run_generation(image_analysis_results, section_data['full_text'], section_title)
            
            if os.path.exists("Generated_Test_Cases_Output.xlsx"):
                os.rename("Generated_Test_Cases_Output.xlsx", output_filename)
        finally:
            os.chdir(original_dir)
        
        job["status"] = "completed"
        job["output_file"] = output_path
        
        return ProcessingResult(
            pdf_id=pdf_id,
            section_title=section_title,
            status="completed",
            text_chunks_count=len(section_data['text_chunks']),
            image_count=section_data['image_count'],
            output_file=output_filename
        )
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return ProcessingResult(
            pdf_id=pdf_id,
            section_title=section_title,
            status="error",
            text_chunks_count=0,
            image_count=0,
            error=str(e)
        )

@app.post("/api/process-all-subsections", response_model=Dict[str, Any])
async def process_all_subsections(request: ProcessAllSubsectionsRequest):
    """
    Process all subsections within a parent section
    
    Returns results for each subsection processed
    """
    pdf_id = request.pdf_id
    section_title = request.section_title
    
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    
    if job["status"] != "sections_extracted":
        raise HTTPException(
            status_code=400,
            detail=f"Job is in '{job['status']}' state. Extract sections first."
        )
    
    try:
        collection = job["collection"]
        
        # Retrieve section with subsections
        section_info = ssp.get_section_data_with_images(
            collection=collection,
            section_title=section_title,
            include_subsections=True
        )
        
        if not section_info['subsections']:
            raise HTTPException(
                status_code=400,
                detail=f"No subsections found for section '{section_title}'"
            )
        
        results = {
            "pdf_id": pdf_id,
            "parent_section": section_title,
            "total_subsections": len(section_info['subsections']),
            "results": []
        }
        
        # Process each subsection
        for idx, subsection in enumerate(section_info['subsections'], 1):
            subsection_title = subsection['section_title']
            
            try:
                # Process images
                image_analysis_results = []
                if subsection['images']:
                    for image_path in subsection['images']:
                        if os.path.exists(image_path):
                            try:
                                analysis = run_analysis(image_path)
                                image_analysis_results.append(str(analysis))
                            except Exception as e:
                                print(f"Error analyzing image {image_path}: {e}")
                
                # Generate test cases
                output_filename = f"TestCases_{pdf_id}_subsection_{idx}.xlsx"
                output_path = os.path.join(job["job_dir"], output_filename)
                
                original_dir = os.getcwd()
                os.chdir(job["job_dir"])
                
                try:
                    run_generation(image_analysis_results, subsection['full_text'], subsection_title)
                    
                    if os.path.exists("Generated_Test_Cases_Output.xlsx"):
                        os.rename("Generated_Test_Cases_Output.xlsx", output_filename)
                finally:
                    os.chdir(original_dir)
                
                results["results"].append({
                    "subsection": subsection_title,
                    "status": "completed",
                    "text_chunks": len(subsection['text_chunks']),
                    "images": len(subsection['images']),
                    "output_file": output_filename
                })
                
            except Exception as e:
                results["results"].append({
                    "subsection": subsection_title,
                    "status": "error",
                    "error": str(e)
                })
        
        job["status"] = "completed"
        return results
        
    except Exception as e:
        job["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# =============================================================================
# NEW FIGMA API ENDPOINTS
# =============================================================================

@app.post("/api/link-figma-project")
async def link_figma_project(request: FigmaProjectRequest):
    """
    Link a Figma project to get prototype flows
    
    Fetches Figma file JSON and extracts prototype navigation flows
    """
    if not request.figma_token:
        raise HTTPException(status_code=400, detail="Figma token is required")
    
    try:
        flows, flows_summary = fetch_figma_flows(request.figma_file_id, request.figma_token)
        
        if not flows:
            raise HTTPException(
                status_code=400,
                detail="No flows found or invalid Figma file ID"
            )
        
        return {
            "status": "success",
            "figma_file_id": request.figma_file_id,
            "total_flows": flows_summary.get('total_flows', 0),
            "screens": flows_summary.get('screens', []),
            "interactions": flows_summary.get('interactions', []),
            "sample_flows": flows_summary.get('sample_flows', []),
            "message": f"Successfully extracted {len(flows)} prototype flows from Figma"
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Figma data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Figma data: {str(e)}")

@app.post("/api/fetch-figma-images")
async def fetch_figma_images_endpoint(request: FigmaProjectRequest):
    """
    Fetch and download images from Figma file
    
    Downloads frame screenshots from Figma for use as context
    """
    try:
        job_dir = "figma_temp"
        Path(job_dir).mkdir(exist_ok=True)
        
        downloaded_images = download_figma_images(
            request.figma_file_id,
            request.figma_token,
            job_dir,
            limit=10
        )
        
        if not downloaded_images:
            raise HTTPException(status_code=400, detail="No images could be downloaded from Figma")
        
        return {
            "status": "success",
            "figma_file_id": request.figma_file_id,
            "downloaded_count": len(downloaded_images),
            "images": downloaded_images,
            "message": f"Downloaded {len(downloaded_images)} images from Figma"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/figma-flows/{figma_file_id}")
async def get_figma_flows(
    figma_file_id: str,
    figma_token: str = Query(...)
):
    """
    Get extracted Figma prototype flows without PDF processing
    
    Returns all prototype flows and screens from a Figma file
    """
    try:
        flows, flows_summary = fetch_figma_flows(figma_file_id, figma_token)
        
        if not flows:
            raise HTTPException(status_code=400, detail="Invalid Figma file ID or no flows found")
        
        # Group flows by screen
        flows_by_screen = {}
        for flow in flows:
            screen = flow['from_screen']
            if screen not in flows_by_screen:
                flows_by_screen[screen] = []
            flows_by_screen[screen].append(flow)
        
        return {
            "figma_file_id": figma_file_id,
            "total_flows": len(flows),
            "screens": list(flows_by_screen.keys()),
            "flows_by_screen": flows_by_screen
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

class ProcessSectionWithFigmaRequest(BaseModel):
    pdf_id: str
    section_number: int
    figma_file_id: str
    figma_token: str

@app.post("/api/process-section-with-figma")
async def process_section_with_figma(request: ProcessSectionWithFigmaRequest):
    """
    Process PDF section with enhanced Figma prototype context
    """
    pdf_id = request.pdf_id
    section_number = request.section_number
    figma_file_id = request.figma_file_id
    figma_token = request.figma_token
    
    job = load_or_recreate_job(pdf_id)
    if not job:
        raise HTTPException(status_code=404, detail="PDF ID not found")

    if section_number not in job["section_map"]:
        raise HTTPException(status_code=400, detail="Invalid section number")

    try:
        section_title = job["section_map"][section_number]['title']
        collection = job["collection"]

        # Get PDF section data
        print(f"📄 Fetching PDF section: {section_title}")
        section_data = ssp.get_section_data_with_images(
            collection=collection,
            section_title=section_title,
            include_subsections=True
        )

        # Fetch Figma flows
        print(f"🎨 Fetching Figma flows from {figma_file_id}")
        flows, flows_summary = fetch_figma_flows(figma_file_id, figma_token)

        # Download Figma images
        print(f"📥 Downloading Figma UI screenshots")
        figma_images = download_figma_images(
            figma_file_id,
            figma_token,
            job["job_dir"],
            limit=10
        )

        # CRITICAL: Convert relative paths to absolute paths
        print(f"🔧 Converting image paths to absolute...")
        figma_images_absolute = []
        for img_path in figma_images:
            if os.path.isabs(img_path):
                abs_path = img_path
            else:
                abs_path = os.path.abspath(img_path)
            
            # Verify file exists
            if os.path.exists(abs_path):
                figma_images_absolute.append(abs_path)
                print(f"   ✓ {os.path.basename(abs_path)} (verified)")
            else:
                print(f"   ✗ {os.path.basename(abs_path)} (NOT FOUND at {abs_path})")

        print(f"✓ Total Figma images ready: {len(figma_images_absolute)}")

        # Create Figma context
        figma_context = create_figma_context_string(flows, flows_summary)

        # Combine contexts
        pdf_context = f"PDF Section: {section_title}\n\nContent:\n{section_data.get('full_text', '')}"
        combined_context = f"{pdf_context}\n\n{figma_context}"

        # Combine images
        all_images = section_data.get('images', []) + figma_images_absolute

        # Generate test cases
        job_dir_absolute = os.path.abspath(job["job_dir"])
        output_filename = f"TestCases_{pdf_id}_{section_title}_with_figma.xlsx"
        output_path = os.path.join(job_dir_absolute, output_filename)

        original_dir = os.getcwd()

        try:
            os.chdir(job_dir_absolute)

            print(f"🧪 Generating test cases with combined context")
            print(f"   Working dir: {os.getcwd()}")
            print(f"   Images to analyze: {len(all_images)}")
            print(f"   Images:")
            for img in all_images:
                exists = os.path.exists(img)
                print(f"      {'✓' if exists else '✗'} {img}")

            # Call test case generation with absolute paths
            run_generations_2(
                section_title,
                all_images,  # Now contains absolute paths
                combined_context
            )

            # Find and move generated file
            generated_file = os.path.join(job_dir_absolute, "Generated_Test_Cases_multi_Pydantic.xlsx")

            if os.path.exists(generated_file):
                print(f"✓ Found generated file")

                if generated_file != output_path:
                    import shutil
                    shutil.move(generated_file, output_path)
                    print(f"✓ Renamed to {output_filename}")
            else:
                # Fallback: search for xlsx files
                try:
                    files_in_dir = os.listdir(job_dir_absolute)
                    xlsx_files = [f for f in files_in_dir if f.endswith('.xlsx')]

                    if xlsx_files:
                        print(f"⚠️  Using {xlsx_files[0]}")
                        found_file = os.path.join(job_dir_absolute, xlsx_files[0])
                        import shutil
                        shutil.move(found_file, output_path)
                    else:
                        raise FileNotFoundError("No Excel files found")
                except Exception as e:
                    raise FileNotFoundError(f"Failed to find generated file: {str(e)}")

        finally:
            os.chdir(original_dir)

        # Verify file exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found at {output_path}")

        print(f"✅ SUCCESS: Test cases generated with Figma context")

        return {
            "pdf_id": pdf_id,
            "section_title": section_title,
            "figma_file_id": figma_file_id,
            "status": "completed",
            "pdf_text_chunks": len(section_data.get('text_chunks', [])),
            "pdf_images_count": len(section_data.get('images', [])),
            "figma_flows_count": len(flows),
            "figma_images_count": len(figma_images_absolute),
            "total_context_images": len(all_images),
            "output_file": output_filename,
            "figma_context": {
                "screens": flows_summary.get('screens', []),
                "interactions": flows_summary.get('interactions', []),
                "total_flows": flows_summary.get('total_flows', 0)
            }
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# =============================================================================
# FIGMA WORKFLOW GENERATOR ENDPOINTS (NEW)
# =============================================================================

@app.post("/api/figma/init", response_model=FigmaSectionsListResponse)
async def figma_init_job(req: FigmaExtractSectionsRequest):
    """
    Initialize a new Figma job and extract sections
    
    Returns: job_id to use in subsequent API calls
    """
    print("------------------------------------------------------------")
    print("INITIALIZE FIGMA JOB - EXTRACT SECTIONS")
    print("------------------------------------------------------------")
    print(f"file_key: {req.file_key}, use_local: {req.use_local}")

    try:
        job_id = str(uuid.uuid4())
        job_dir = create_figma_job_directory(job_id)
        
        token = req.figma_token or os.getenv("FIGMA_TOKEN")
        data = get_figma_file(req.file_key, token, req.use_local)

        sections_dict = {}
        extract_figma_sections(data["document"], sections_dict)

        print(f"[FIGMA] Found {len(sections_dict)} section(s)")
        
        job = {
            "job_id": job_id,
            "file_key": req.file_key,
            "status": "sections_extracted",
            "job_dir": job_dir,
            "sections": sections_dict,
            "use_local": req.use_local,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "output_file": None
        }
        
        figma_jobs[job_id] = job
        save_figma_job_metadata(job_id, job)
        
        sections_list = [
            FigmaSectionInfo(
                section_id=sid,
                section_name=sdata["name"],
                type=sdata["type"]
            )
            for sid, sdata in sections_dict.items()
        ]
        
        print(f"✅ Figma job initialized: {job_id}")
        
        return FigmaSectionsListResponse(
            job_id=job_id,
            file_key=req.file_key,
            total_sections=len(sections_list),
            sections=sections_list
        )

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/figma/generate-test-cases")
async def figma_generate_test_cases(req: FigmaGenerateTestCasesRequest):
    """
    Generate test cases for a Figma workflow section
    """
    print("------------------------------------------------------------")
    print("GENERATE FIGMA TEST CASES")
    print("------------------------------------------------------------")
    print(f"file_key   : {req.file_key}")
    print(f"section_id : {req.section_id}")
    print(f"section    : {req.section_name}")
    print(f"use_local  : {req.use_local}")

    try:
        job_id = str(uuid.uuid4())
        job_dir = create_figma_job_directory(job_id)
        
        token = req.figma_token or os.getenv("FIGMA_TOKEN")

        # Validate Figma file exists
        if req.use_local:
            _ = load_figma_local_json(req.file_key)
            json_source = "local_json"
        else:
            if not token:
                raise HTTPException(400, "Missing Figma token for API request")
            _ = get_figma_file(req.file_key, token, False)
            json_source = "figma_api"

        # FIX: Load image from mock directory based on FIGMA_MODE
        figma_mode = os.getenv("FIGMA_MODE", "mock").lower()
        
        if figma_mode == "mock":
            # Load from local mock images directory
            section_filename = req.section_name.lower().replace(" ", "_").replace("/", "_")
            mock_image_path = f"figma_mocks/images/{req.file_key}/{section_filename}.png"
            
            print(f"[FIGMA] Loading mock image from: {mock_image_path}")
            
            if not os.path.exists(mock_image_path):
                raise HTTPException(
                    400,
                    f"Mock image not found at {mock_image_path}. "
                    f"Expected: figma_mocks/images/{req.file_key}/{section_filename}.png"
                )
            
            with open(mock_image_path, 'rb') as f:
                img_bytes = f.read()
            
            if len(img_bytes) < 100:
                raise HTTPException(400, f"Image too small ({len(img_bytes)} bytes) - invalid file")
            
            img_b64 = base64.b64encode(img_bytes).decode()
            print(f"[FIGMA] Mock image loaded: {len(img_bytes)} bytes")
        else:
            # Download from Figma API in live mode
            print("[FIGMA] Downloading workflow diagram images from Figma API...")
            figma_images = download_figma_images(
                req.file_key,
                token,
                job_dir,
                limit=1
            )

            if not figma_images:
                raise HTTPException(
                    400, 
                    "No images found in Figma file. Make sure file has exportable frames."
                )

            image_path = figma_images[0]
            print(f"[FIGMA] Using image: {image_path}")

            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            
            if len(img_bytes) < 100:
                raise HTTPException(400, f"Downloaded image is invalid or too small ({len(img_bytes)} bytes)")
            
            img_b64 = base64.b64encode(img_bytes).decode()
            print(f"[FIGMA] Image size: {len(img_bytes)} bytes")

        # Generate positive test cases
        print("[FIGMA] Generating positive test cases...")
        pos_cases, pos_ms = await generate_figma_cases(
            img_b64,
            req.section_name,
            "POSITIVE",
            req.google_api_key
        )

        # Generate negative test cases
        print("[FIGMA] Generating negative test cases...")
        neg_cases, neg_ms = await generate_figma_cases(
            img_b64,
            req.section_name,
            "NEGATIVE",
            req.google_api_key
        )

        # Save Excel file
        output_filename = save_figma_excel(pos_cases, neg_cases, job_dir, req.section_name)
        
        # Save job metadata
        job = {
            "job_id": job_id,
            "file_key": req.file_key,
            "section_id": req.section_id,
            "section_name": req.section_name,
            "status": "completed",
            "job_dir": job_dir,
            "use_local": req.use_local,
            "output_file": output_filename,
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "positive_count": len(pos_cases),
            "negative_count": len(neg_cases),
            "llm_positive_ms": pos_ms,
            "llm_negative_ms": neg_ms,
            "json_source": json_source,
            "figma_mode": figma_mode,
        }
        
        figma_jobs[job_id] = job
        save_figma_job_metadata(job_id, job)

        print("[FIGMA] Test case generation complete ✓")
        
        return {
            "job_id": job_id,
            "status": "success",
            "file_key": req.file_key,
            "section_name": req.section_name,
            "file_name": output_filename,
            "positive_count": len(pos_cases),
            "negative_count": len(neg_cases),
            "details": {
                "json_source": json_source,
                "figma_mode": figma_mode,
                "llm_positive_ms": pos_ms,
                "llm_negative_ms": neg_ms,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/api/figma/job-status/{job_id}", response_model=FigmaJobStatus)
async def figma_get_job_status(job_id: str):
    """Get status of a Figma job"""
    job = load_or_recreate_figma_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Figma job ID not found")
    
    return FigmaJobStatus(
        job_id=job_id,
        file_key=job.get("file_key"),
        status=job.get("status"),
        sections_count=len(job.get("sections", {})) if job.get("sections") else None,
        current_section=job.get("section_name"),
        output_file=job.get("output_file"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at")
    )


@app.get("/api/figma/download-results/{job_id}/{filename}")
async def figma_download_results(job_id: str, filename: str):
    """Download generated Figma test cases Excel file"""
    job = load_or_recreate_figma_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Figma job ID not found")
    
    file_path = Path(job["job_dir"]) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.delete("/api/figma/cleanup/{job_id}")
async def figma_cleanup_job(job_id: str):
    """Clean up Figma job resources and files"""
    job = load_or_recreate_figma_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Figma job ID not found")
    
    if job_id in figma_jobs:
        del figma_jobs[job_id]
    
    return {
        "message": f"Figma job {job_id} cleaned up successfully",
        "job_id": job_id
    }


@app.get("/api/job-status/{pdf_id}", response_model=JobStatus)
async def get_job_status(pdf_id: str):
    """Get the current status of a processing job"""
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    
    # Get processed PDF info if available
    pdf_info = get_pdf_by_id(pdf_id)
    
    return JobStatus(
        pdf_id=pdf_id,
        status=job["status"],
        message=f"Job is in '{job['status']}' state",
        sections_count=len(job["sections"]) if job.get("sections") else None,
        current_step=job.get("current_section"),
        pdf_file=job.get("pdf_filename"),
        processed_at=pdf_info.get('processed_at') if pdf_info else None
    )

@app.get("/api/sections/{pdf_id}", response_model=SectionsListResponse)
async def get_sections(pdf_id: str):
    """Get the list of extracted sections for a PDF"""
    job = load_or_recreate_job(pdf_id)
    if not job:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    if not job.get("sections") or not job.get("section_map"):
        raise HTTPException(
            status_code=400,
            detail="Sections not yet extracted. Call /extract-sections first."
        )
    
    print(job)
    
    # sections_info = [
    #     SectionInfo(
    #         number=num,
    #         level=section_data["level"],
    #         title=section_data["title"],
    #         parent_section_title=section_data.get("parent_section_title"),
    #         indexed_pages=section_data["indexed_pages"],
    #         image_count=section_data["image_count"],
    #         section_number = section_data['section_number']
    #     )
    #     for num, section_data in job["section_map"].items()
    # ]

    sections_info = []

    for num, section_data in job["section_map"].items():
        print("\n------ ITERATION ------")
        print("num:", num)
        print("section_data:", section_data)

        section = SectionInfo(
            number=num,
            level=section_data["level"],
            title=section_data["title"],
            parent_section_title=section_data.get("parent_section_title"),
            indexed_pages=section_data["indexed_pages"],
            image_count=section_data["image_count"],
            section_number=section_data["section_number"]
        )
        sections_info.append(section)

    print(sections_info)
    
    return SectionsListResponse(
        pdf_id=pdf_id,
        total_sections=len(job["sections"]),
        sections=sections_info
    )

@app.get("/api/processed-pdfs", response_model=PDFListResponse)
async def list_processed_pdfs():
    """List all previously processed PDFs with their IDs"""
    pdfs_metadata = list_all_processed_pdfs()
    
    pdfs_list = [
        ProcessedPDFInfo(
            pdf_id=pdf_id,
            pdf_file=info['pdf_file'],
            pdf_path=info['pdf_path'],
            processed_at=info['processed_at'],
            sections_count=info['sections_count'],
            total_images=info['total_images'],
            status=info['status']
        )
        for pdf_id, info in pdfs_metadata.items()
    ]
    
    return PDFListResponse(
        total_pdfs=len(pdfs_list),
        pdfs=pdfs_list
    )

@app.get("/api/processed-pdf/{pdf_id}", response_model=ProcessedPDFInfo)
async def get_processed_pdf_info(pdf_id: str):
    """Get information about a previously processed PDF using its ID"""
    pdf_info = get_pdf_by_id(pdf_id)
    
    if not pdf_info:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    return ProcessedPDFInfo(
        pdf_id=pdf_id,
        pdf_file=pdf_info['pdf_file'],
        pdf_path=pdf_info['pdf_path'],
        processed_at=pdf_info['processed_at'],
        sections_count=pdf_info['sections_count'],
        total_images=pdf_info['total_images'],
        status=pdf_info['status']
    )

@app.get("/api/download-results/{pdf_id}/{filename}")
@app.head("/api/download-results/{pdf_id}/{filename}") 
async def download_results(pdf_id: str, filename: str):
    """Download the generated test cases Excel file"""
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    file_path = os.path.join(job["job_dir"], filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.delete("/api/cleanup/{pdf_id}")
async def cleanup_job(pdf_id: str):
    """Clean up job resources and files"""
    if pdf_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="PDF ID not found")
    
    job = processing_jobs[pdf_id]
    
    # Delete ChromaDB collection
    if job.get("collection_name"):
        try:
            chroma_client.delete_collection(name=job["collection_name"])
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    # Remove from tracking
    del processing_jobs[pdf_id]
    
    return {
        "message": f"Job {pdf_id} cleaned up successfully",
        "pdf_id": pdf_id
    }

# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    Path("jobs").mkdir(exist_ok=True)
    print("✅ FastAPI server started successfully")
    print(f"✅ Total processed PDFs: {len(list_all_processed_pdfs())}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 FastAPI server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)