import os
import fitz  # PyMuPDF
from typing import List, Dict

def detect_scanned_pdf(page) -> bool:
    """
    Detects if a page is likely scanned by checking the presence of text.
    If text length is extremely low compared to the page size, it's likely an image.
    """
    text = page.get_text().strip()
    # Simple heuristic: if text length is less than 50 chars, it might be a scanned image
    # or just a page with very little content.
    return len(text) < 50

def load_pdf_with_pymupdf(file_path: os.PathLike) -> List[Dict]:
    """
    Loads a PDF document using PyMuPDF and extracts page-wise text.
    Detects scanned pages.
    """
    doc_data = []
    file_name = os.path.basename(file_path)
    
    print(f"Opening PDF with PyMuPDF: {file_name}")
    
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            is_scanned = detect_scanned_pdf(page)
            
            page_data = {
                "content": text,
                "metadata": {
                    "source": file_name,
                    "page": page_num + 1,
                    "is_scanned": is_scanned,
                    "total_pages": len(doc)
                }
            }
            doc_data.append(page_data)
            
            if is_scanned:
                print(f"  [Warning] Page {page_num + 1} looks like a scanned image/low text.")
        
        doc.close()
    except Exception as e:
        print(f"  [Error] Failed to process {file_name}: {e}")
        
    return doc_data

def load_document(file_path: os.PathLike) -> List[Dict]:
    """
    Main entry for loading documents.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf_with_pymupdf(file_path)
    elif ext == ".txt":
        # Handle TXT similarly for consistency
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{
            "content": content,
            "metadata": {"source": os.path.basename(file_path), "page": 1}
        }]
    else:
        print(f"Unsupported format: {ext}")
        return []
