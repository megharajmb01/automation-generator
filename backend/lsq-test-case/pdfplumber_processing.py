import pdfplumber

def open_pdf_file_using_plumber(pdf_path):
    pdf_plumber = pdfplumber.open(pdf_path)
    return(pdf_plumber)

def extract_non_table_text_from_page(pdfplumber, page_index):
    """
    Extracts text from a single pdfplumber page, excluding text within tables.
    """
    # Tables, Images, and Attachments are ignored by default by the 'chars' method,
    # but we need to explicitly filter text based on table bounding boxes.

    # 1. Find all table bounding boxes on the page
    page = pdfplumber.pages[page_index]
    tables = page.find_tables()
    table_rects = [t.bbox for t in tables]

    non_table_chars = []

    # 2. Iterate through every character and check if it's inside any table box
    for char in page.chars:
        is_in_table = False

        for t_rect in table_rects:
            # Check if the character's bounding box is fully contained within the table's bbox
            if (t_rect[0] <= char["x0"] and char["x1"] <= t_rect[2] and
                    t_rect[1] <= char["top"] and char["bottom"] <= t_rect[3]):
                is_in_table = True
                break

        if not is_in_table:
            non_table_chars.append(char)

    # 3. Reconstruct the text from the non-table characters
    # This reconstruction method is simple but often effective for clean text.
    return "".join(c["text"] for c in non_table_chars)
        # No attachments or raw image data will be processed by these calls.