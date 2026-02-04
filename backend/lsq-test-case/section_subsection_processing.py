from pypdf import PdfReader
from pypdf.generic import Destination
import chromadb
import re
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path
from datetime import datetime
import os

import pdfplumber_processing
from typing import List, Dict, Optional, Tuple
from chromadb import PersistentClient
from image_processing import run_analysis
from test_case_generation_multimodal import run_generations_2


COLLECTION_NAME = "pdf_section_rag_index"
PROCESSING_METADATA_FILE = "pdf_processing_metadata.json"  # Track processed PDFs

def get_image_output_folder(pdf_path):
    job_dir = os.path.dirname(os.path.abspath(pdf_path))
    output_folder = os.path.join(job_dir, "extracted_images_by_section")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    return output_folder

def get_collection_name_for_pdf(pdf_id: str) -> str:
    """Get the ChromaDB collection name for a specific PDF"""
    return f"pdf_{pdf_id}"

# ============================================================
# PDF PROCESSING METADATA MANAGEMENT
# ============================================================

def generate_pdf_unique_id():
    """Generate a unique ID for each PDF processing session"""
    return str(uuid.uuid4())

def get_pdf_file_hash(pdf_path):
    """Generate hash of PDF file for validation"""
    hash_md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processing_metadata():
    """Load metadata about processed PDFs"""
    if os.path.exists(PROCESSING_METADATA_FILE):
        with open(PROCESSING_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_processing_metadata(metadata):
    """Save metadata about processed PDFs"""
    with open(PROCESSING_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def sanitize_section_name(name: str):
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name)

# ============================================================
# IMAGE FILTERING HELPER FUNCTIONS
# ============================================================

def is_small_image(width, height, min_width=0, min_height=0):
    """Check if image is too small (likely a logo/icon)"""
    return width < min_width or height < min_height

def is_header_position(y_position, page_height, header_threshold=0.10):
    """Check if image is in header area (top 10% of page)"""
    return y_position < (page_height * header_threshold)

def is_footer_position(y_position, page_height, footer_threshold=0.85):
    """Check if image is in footer area (bottom 15% of page)"""
    return y_position > (page_height * footer_threshold)

def filter_duplicate_images(image_hashes, duplicate_threshold=3):
    """Identify images that appear multiple times (likely logos/headers)"""
    hash_counts = Counter(image_hashes)
    duplicates = {hash_val for hash_val, count in hash_counts.items() 
                  if count >= duplicate_threshold}
    return duplicates

# ============================================================
# CHROMA DB RETRIEVAL HELPERS
# ============================================================

def debug_section_in_collection(collection, section_title):
    """
    Debug function to see what's actually stored for a section
    """
    print(f"\n🔍 DEBUG: Checking collection for section: {section_title}")
    
    # Check main section
    main_results = collection.get(
        where={"section_title": section_title},
        limit=5,
        include=['documents', 'metadatas']
    )
    
    print(f"   Main section chunks: {len(main_results['documents'])}")
    if main_results['metadatas']:
        print(f"   First chunk metadata keys: {main_results['metadatas'][0].keys()}")
        if 'image_paths' in main_results['metadatas'][0]:
            print(f"   Image paths in first chunk: {main_results['metadatas'][0]['image_paths']}")
    
    # Check subsections
    sub_results = collection.get(
        where={"parent_section_title": section_title},
        limit=100,
        include=['documents', 'metadatas']
    )
    
    print(f"   Subsection chunks found: {len(sub_results['documents'])}")
    
    # Group by section
    if sub_results['metadatas']:
        sections_found = {}
        for meta in sub_results['metadatas']:
            sect_title = meta.get('section_title', 'Unknown')
            if sect_title not in sections_found:
                sections_found[sect_title] = {
                    'chunks': 0,
                    'has_images': False,
                    'image_count': 0
                }
            sections_found[sect_title]['chunks'] += 1
            if meta.get('chunk_index') == 0 and 'image_paths' in meta and meta['image_paths']:
                sections_found[sect_title]['has_images'] = True
                sections_found[sect_title]['image_count'] = len(meta['image_paths'].split(';'))
        
        for sect, info in sections_found.items():
            print(f"      • {sect}: {info['chunks']} chunks, Images: {info['image_count']}")

def get_section_data_with_images(collection, section_title, section_number, include_subsections=True):
    """
    Retrieves complete section data including text chunks and image paths.
    FIXED: Properly collects images from all subsections.
    """
    if include_subsections:
        results = collection.get(
            where={"section_number": section_number},
            limit=10000,
            include=['documents', 'metadatas']
        )
    else:
        results = collection.get(
            where={"section_title": section_title},
            limit=10000,
            include=['documents', 'metadatas']
        )
    
    if not results['documents']:
        return {
            'section_title': section_title,
            'text_chunks': [],
            'full_text': '',
            'images': [],
            'metadata': {},
            'has_images': False,
            'subsections': []
        }
    
    text_chunks = results['documents']
    metadatas = results['metadatas']
    
    # Collect images from the main section (first chunk only)
    image_paths = []
    first_metadata = metadatas[0] if metadatas else {}
    
    if 'image_paths' in first_metadata and first_metadata['image_paths']:
        image_paths = first_metadata['image_paths'].split(';')
    
    has_images = first_metadata.get('has_images', 'false') == 'true'
    full_text = '\n\n'.join(text_chunks)
    
    subsections_data = []
    if include_subsections:
        section_groups = {}
        for i, metadata in enumerate(metadatas):
            current_section = metadata.get('section_title', '')
            if current_section not in section_groups:
                section_groups[current_section] = {
                    'section_title': current_section,
                    'level': metadata.get('level', 0),
                    'parent_section_title': metadata.get('parent_section_title', ''),
                    'section_number': metadata.get('section_number', ''),
                    'chunks': [],
                    'images': [],
                    'metadata': metadata
                }
            section_groups[current_section]['chunks'].append(text_chunks[i])
            
            # FIXED: Collect images from first chunk of each section
            if metadata.get('chunk_index', -1) == 0:
                if 'image_paths' in metadata and metadata['image_paths']:
                    sub_images = metadata['image_paths'].split(';')
                    section_groups[current_section]['images'] = sub_images
                    print(f"   🖼️  Found {len(sub_images)} images in section: {current_section}")
        
        for sect_title, sect_data in section_groups.items():
            if sect_title != section_title:  # Exclude the main section itself
                subsections_data.append({
                    'section_title': sect_data['section_title'],
                    'level': sect_data['level'],
                    'parent_section_title': sect_data['parent_section_title'],
                    'section_number': sect_data['section_number'],
                    'text_chunks': sect_data['chunks'],
                    'full_text': '\n\n'.join(sect_data['chunks']),
                    'images': sect_data['images'],
                    'has_images': len(sect_data['images']) > 0
                })
    
    return {
        'section_title': section_title,
        'text_chunks': text_chunks,
        'full_text': full_text,
        'images': image_paths,
        'metadata': first_metadata,
        'has_images': has_images,
        'image_count': len(image_paths),
        'subsections': subsections_data
    }

def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple text splitter for demonstration purposes."""
    if not text:
        return []
    if len(text) > chunk_size:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        return chunks
    else:
        return [text]

# ============================================================
# FLATTEN PYPDF OUTLINE
# ============================================================

def get_flat_pypdf_outline(items: List, level: int = 1, parent_numbering: str = "") -> List[Dict]:
    """
    Recursively flattens the nested pypdf outline structure and adds
    a calculated 'section_number' (e.g., '1.2.1').
    """
    flat_list = []

    # Initialize the counter for the current level (e.g., 1, 2, 3...)
    current_level_counter = 1

    for item in items:
        if isinstance(item, Destination):
            # 1. Calculate the full section number for the current item
            # If it's a top level (level 1), parent_numbering is ""
            if parent_numbering:
                section_number = f"{parent_numbering}.{current_level_counter}"
            else:
                section_number = str(current_level_counter)

            # 2. Add the item and its number to the flat list

            flat_list.append({
                "level": level,
                "title": item.title.strip(),
                "destination": item,
                "section_number": section_number
            })

            # 3. Increment the counter for the current level
            current_level_counter += 1

        elif isinstance(item, list):
            # If it's a list (a nested outline structure), recursively call
            # the function, using the number of the *previous* major section
            # element (the one just processed) as the new parent_numbering.

            # The last item added to flat_list must be the parent of this nested list.
            if flat_list and "section_number" in flat_list[-1]:
                new_parent_numbering = flat_list[-1]["section_number"]
            else:
                # Fallback, though this should ideally not be hit with a well-formed TOC
                new_parent_numbering = ""

            flat_list.extend(
                get_flat_pypdf_outline(
                    item,
                    level + 1,
                    new_parent_numbering  # <- Pass the new parent number
                )
            )

    return flat_list

# ============================================================
# ANALYZE ALL IMAGES FOR DUPLICATES
# ============================================================

def analyze_duplicate_images(pdf_plumber):
    """First pass: analyze all images to identify duplicates (logos/headers)"""
    image_hashes = []
    
    print("🔍 Pass 1: Analyzing images for duplicates...")
    for page in pdf_plumber.pages:
        for img in page.images:
            width = img['x1'] - img['x0']
            height = img['bottom'] - img['top']
            img_signature = f"{width}_{height}_{img.get('name', '')}"
            image_hashes.append(img_signature)
    
    duplicate_hashes = filter_duplicate_images(image_hashes, duplicate_threshold=3)
    print(f"✅ Found {len(duplicate_hashes)} duplicate image patterns")
    print("-" * 60)
    
    return duplicate_hashes

# ============================================================
# NEW: GET PRIMARY SECTION FOR PAGE
# ============================================================

def get_primary_section_for_page(page_idx, page_to_sections):
    """
    Returns the most specific (deepest level) section for a given page.
    When multiple sections start on the same page:
    1. Prefer sections with higher level (deeper nesting)
    2. If same level, prefer the FIRST one (earliest in document order using 'order' field)
    """
    if page_idx not in page_to_sections:
        return None
    
    sections = page_to_sections[page_idx]
    if not sections:
        return None
    
    # Find the maximum level
    max_level = max(s["level"] for s in sections)
    
    # Get all sections at the maximum level
    sections_at_max_level = [s for s in sections if s["level"] == max_level]
    
    # Sort by order (document sequence) and return the first one
    sections_at_max_level.sort(key=lambda x: x["order"])
    
    return sections_at_max_level[0]

# ============================================================
# IMAGE EXTRACTION FOR SECTION
# ============================================================

def extract_images_for_section(pdf_plumber, section_title, section_number,
                               parent_section_number, start_page, end_page, 
                               duplicate_hashes, pdf_path,
                               filter_logos=True, min_size=0, 
                               exclude_header_footer=True,
                               page_to_sections=None):
    """
    Extract images that ACTUALLY BELONG to this section.
    Uses the most specific section when multiple sections share a page.
    """
    
    if page_to_sections is None:
        page_to_sections = {}
    
    job_dir = os.path.dirname(os.path.abspath(pdf_path))
    output_base = os.path.join(job_dir, "extracted_images_by_section")
    
    section_folder_name = f"{section_number}_{sanitize_section_name(section_title)}"
    
    if parent_section_number:
        section_parts = section_number.split('.')
        path_parts = [output_base]
        
        for j in range(len(section_parts) - 1):
            partial_number = '.'.join(section_parts[:j+1])
            path_parts.append(f"{partial_number}_")
        
        path_parts.append(section_folder_name)
        section_folder = os.path.join(*path_parts)
    else:
        section_folder = os.path.join(output_base, section_folder_name)
    
    Path(section_folder).mkdir(parents=True, exist_ok=True)
    
    extracted_images = []
    
    for page_index in range(start_page, end_page + 1):
        if page_index >= len(pdf_plumber.pages):
            break
        
        # CHECK: Does this page actually belong to this section?
        if page_index in page_to_sections:
            primary_section = get_primary_section_for_page(page_index, page_to_sections)
            if primary_section and primary_section["section_number"] != section_number:
                # This page's primary section is different, skip it
                print(f"    ⏭️  Skipping page {page_index + 1} (belongs to section {primary_section['section_number']}, not {section_number})")
                continue
        
        page = pdf_plumber.pages[page_index]
        images = page.images
        page_height = page.height
        
        for img_index, img in enumerate(images, start=1):
            try:
                width = img['x1'] - img['x0']
                height = img['bottom'] - img['top']
                y_position = img['top']
                
                img_signature = f"{width}_{height}_{img.get('name', '')}"
                
                skip_reason = None
                
                if filter_logos and img_signature in duplicate_hashes:
                    skip_reason = "duplicate/logo"
                elif is_small_image(width, height, min_size, min_size):
                    skip_reason = f"too small ({width:.0f}x{height:.0f})"
                elif exclude_header_footer and is_header_position(y_position, page_height):
                    skip_reason = "in header area"
                elif exclude_header_footer and is_footer_position(y_position, page_height):
                    skip_reason = "in footer area"
                
                if skip_reason:
                    continue
                
                image = page.within_bbox(
                    (img['x0'], img['top'], img['x1'], img['bottom'])
                ).to_image(resolution=300)
                
                image_filename = f"page{page_index + 1}_img{img_index}.png"
                image_path = os.path.join(section_folder, image_filename)
                
                image.save(image_path, quality=100, optimize=False)
                
                extracted_images.append({
                    "path": image_path,
                    "filename": image_filename,
                    "page": page_index + 1,
                    "dimensions": f"{width:.0f}x{height:.0f}",
                    "position": f"y={y_position:.0f}"
                })
                
                print(f"    📷 Extracted: {image_filename} ({width:.0f}x{height:.0f}) → {section_number}")
                
            except Exception as e:
                print(f"    ✗ Image extraction error: {str(e)}")
    
    return extracted_images

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def get_toc_and_extract_data(pdf_path, collection, extract_images=True,
                             filter_logos=True, min_image_size=0,
                             exclude_header_footer=True):
    """
    Reads the TOC/Outline of a PDF, flattens the structure, and extracts
    the corresponding text data AND images for each section/subsection.
    Images are assigned to the CORRECT section based on page position and section title.
    FIXED: Now properly handles multiple sections on the same page.
    """
    try:
        reader = PdfReader(pdf_path)
        outline = reader.outline
        pdf_plumber = pdfplumber_processing.open_pdf_file_using_plumber(pdf_path)

        if not outline:
            print("❌ Error: PDF does not contain an Outline (Table of Contents).")
            return []

        print("✅ Successfully Read PDF (Table of Contents).")
        
        duplicate_hashes = set()
        if extract_images:
            duplicate_hashes = analyze_duplicate_images(pdf_plumber)
        
        sections_data = []
        total_pages = len(reader.pages)
        
        flat_outline = get_flat_pypdf_outline(outline)

        # Build a mapping: page_number -> LIST of sections
        # This handles multiple sections starting on the same page
        page_to_sections = {}
        
        for i, current_item in enumerate(flat_outline):
            section_number = current_item["section_number"]
            title = current_item["title"].strip()
            destination = current_item["destination"]
            
            start_page = reader.get_destination_page_number(destination)
            
            # Calculate end page
            if i + 1 < len(flat_outline):
                next_destination = flat_outline[i + 1]["destination"]
                next_start = reader.get_destination_page_number(next_destination)
                end_page = max(start_page, next_start - 1)
            else:
                end_page = total_pages - 1
            
            # Map all pages in this section to a LIST of sections
            for page_idx in range(start_page, end_page + 1):
                if page_idx not in page_to_sections:
                    page_to_sections[page_idx] = []
                
                page_to_sections[page_idx].append({
                    "section_number": section_number,
                    "title": title,
                    "level": current_item["level"],
                    "order": i  # Add order to preserve sequence
                })

        # Debug: Print page-to-section mapping for pages with multiple sections
        print("\n🗺️  Page-to-Section Mapping (multiple sections per page):")
        for page_idx in sorted(page_to_sections.keys()):
            sections = page_to_sections[page_idx]
            if len(sections) > 1:
                primary = get_primary_section_for_page(page_idx, page_to_sections)
                print(f"   Page {page_idx + 1}: {len(sections)} sections → Primary: {primary['section_number']} (L{primary['level']}) {primary['title']}")
                for sect in sorted(sections, key=lambda x: x['order']):
                    marker = "✓" if sect['section_number'] == primary['section_number'] else " "
                    print(f"      {marker} [{sect['order']}] {sect['section_number']} (L{sect['level']}): {sect['title']}")
        print("-" * 60)

        # Now process each section
        title = None
        level = -1
        
        for i, current_item in enumerate(flat_outline):
            if i == 0:
                prev_title = None
                prev_title_level = -1
            else:
                prev_title = title
                prev_title_level = level

            title = current_item["title"].strip()
            level = current_item["level"]
            destination = current_item["destination"]
            section_number = current_item["section_number"]

            # Determine hierarchy
            if level == 1:
                parent_section_title = None
                parent_section_number = None
                main_section_title = title
            else:
                parts = section_number.rsplit('.', 1)
                parent_section_number = parts[0] if len(parts) > 1 else None
                parent_section_title = prev_title if prev_title_level < level else None

            start_page_index = reader.get_destination_page_number(destination)
            end_page_index = total_pages - 1

            if i + 1 < len(flat_outline):
                next_item = flat_outline[i + 1]
                next_destination = next_item["destination"]
                next_start = reader.get_destination_page_number(next_destination)
                # If next section starts on same page or next page, end at current page
                end_page_index = max(start_page_index, next_start - 1)

            print(f'Section Details | Section {section_number}| L{level} | Title: "{title}" | Pages: {start_page_index + 1}-{end_page_index + 1}')

            content = ""
            if 0 <= start_page_index < total_pages:
                for page_index in range(start_page_index, min(end_page_index + 1, total_pages)):
                    page = reader.pages[page_index]
                    page_content = page.extract_text()

                    page_content = page_content.replace('ﬂ', 'fl')
                    page_content = re.sub(r'\s+', ' ', page_content)

                    # Find where this section starts in the page
                    start_index = page_content.find(title)
                    if (start_index != -1):
                        page_content = page_content[start_index:]
                    
                    # Remove previous section's title if it appears before current
                    if prev_title and (prev_title in page_content) and (prev_title != title) and (title in page_content):
                        page_content = title + page_content.split(title, 1)[1]

                    # FIXED: Only cut at next section if it's at a LOWER or EQUAL level (sibling/uncle)
                    # Don't cut at child sections (subsections should be included)
                    if i + 1 < len(flat_outline):
                        next_item = flat_outline[i + 1]
                        next_title = next_item["title"].strip()
                        next_level = next_item["level"]
                        
                        # Only cut if next section is NOT a child of current section
                        # (i.e., next level should be <= current level for cutting)
                        if next_title and next_title in page_content:
                            if next_level <= level:
                                # Next section is sibling or parent level - cut here
                                page_content = page_content.split(next_title, 1)[0]
                            # If next_level > level, it's a subsection - keep it in content

                    content += page_content + "\n"

            extracted_images = []
            if extract_images:
                extracted_images = extract_images_for_section(
                    pdf_plumber=pdf_plumber,
                    section_title=title,
                    section_number=section_number,
                    parent_section_number=parent_section_number,
                    start_page=start_page_index,
                    end_page=end_page_index,
                    duplicate_hashes=duplicate_hashes,
                    pdf_path=pdf_path,
                    filter_logos=filter_logos,
                    min_size=min_image_size,
                    exclude_header_footer=exclude_header_footer,
                    page_to_sections=page_to_sections  # <- Pass the mapping (now a list per page)
                )

            full_text = content.strip()

            if full_text:
                chunks = chunk_text_by_tokens(
                    text=full_text,
                    chunk_size=1000,
                    overlap=100
                )

                metadata_base = {
                    "pdf_title": pdf_path,
                    "section_title": title,
                    "level": level,
                    "parent_section_title": parent_section_title if parent_section_title else "",
                    "main_section_title": main_section_title,
                    "page_range": f"{start_page_index + 1}-{end_page_index + 1}",
                    "image_count": len(extracted_images),
                    "has_images": "true" if extracted_images else "false",
                    "section_number": section_number
                }

                chunk_documents = []
                chunk_metadatas = []
                chunk_ids = []

                for k, chunk in enumerate(chunks):
                    chunk_documents.append(chunk)
                    chunk_metadata = metadata_base.copy()
                    chunk_metadata["chunk_index"] = k
                    
                    if k == 0 and extracted_images:
                        chunk_metadata["image_paths"] = ";".join([img["path"] for img in extracted_images])
                    
                    chunk_metadatas.append(chunk_metadata)
                    unique_id = f"{title.replace(' ', '_').replace('/', '-')}_{start_page_index}_{k}"
                    chunk_ids.append(unique_id)

                collection.add(
                    documents=chunk_documents,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )

            del content
            del full_text

            sections_data.append({
                "level": level,
                "title": title,
                "section_number": section_number,
                "parent_section_title": parent_section_title,
                "parent_section_number": parent_section_number,
                "indexed_pages": f"{start_page_index + 1}-{end_page_index + 1}",
                "image_count": len(extracted_images),
                "images": extracted_images,
            })

            print(f"✅ Section {section_number}: {len(extracted_images)} images extracted")
            print("-" * 60)

        print(f"\n📊 Total sections processed: {len(sections_data)}")
        return sections_data

    except FileNotFoundError:
        print(f"❌ Error: File not found at path: {pdf_path}")
        return []
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return []