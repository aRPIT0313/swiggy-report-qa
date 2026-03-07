import re
import json
import pdfplumber


def extract_text_from_pdf(pdf_path):
    """Extract text from each page with metadata."""
    pages = []
    print(f"Loading PDF: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_num": page_num,
                    "raw_text": text,
                    "source": f"Swiggy Annual Report - Page {page_num}"
                })
            if page_num % 20 == 0:
                print(f"Processed {page_num}/{total_pages} pages...")
    
    print(f"Extracted text from {len(pages)} pages")
    return pages


def clean_text(text):
    """Clean PDF text by fixing hyphens, whitespace, and removing noise."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text, page_num, source, chunk_size=600, overlap=100):
    """Split cleaned text into overlapping chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_index = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)
        
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk).strip()
            if len(chunk_text_str) > 50:
                chunks.append({
                    "chunk_id": f"p{page_num}_c{chunk_index}",
                    "text": chunk_text_str,
                    "page_num": page_num,
                    "source": source,
                    "char_count": len(chunk_text_str)
                })
                chunk_index += 1
            
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_len = overlap_len
        
        current_chunk.append(sentence)
        current_len += sentence_len + 1
    
    if current_chunk:
        chunk_text_str = ' '.join(current_chunk).strip()
        if len(chunk_text_str) > 50:
            chunks.append({
                "chunk_id": f"p{page_num}_c{chunk_index}",
                "text": chunk_text_str,
                "page_num": page_num,
                "source": source,
                "char_count": len(chunk_text_str)
            })
    
    return chunks


def process_pdf(pdf_path, chunk_size=600, overlap=100):
    """Extract → clean → chunk → return all chunks."""
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    
    for page in pages:
        cleaned = clean_text(page["raw_text"])
        if not cleaned:
            continue
        chunks = chunk_text(
            text=cleaned,
            page_num=page["page_num"],
            source=page["source"],
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks, output_path):
    """Save chunks to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunks saved to {output_path}")


def load_chunks(chunks_path):
    """Load chunks from JSON."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)