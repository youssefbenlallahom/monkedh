"""
Module for processing and chunking documents for RAG.
"""

from typing import List, Dict, Any
import re


class DocumentChunker:
    """Handles document loading and chunking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separator: Separator to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def load_markdown(self, file_path: str) -> str:
        """
        Load markdown file content.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sections from markdown content based on headers.
        
        Args:
            content: Markdown content
            
        Returns:
            List of section dictionaries with title and content
        """
        sections = []
        
        # Split by headers (##, ###, etc.)
        parts = re.split(r'\n(#{1,3}\s+.+)\n', content)
        
        current_section = {
            "title": "Introduction",
            "content": "",
            "level": 0
        }
        
        for i, part in enumerate(parts):
            if part.startswith('#'):
                # This is a header
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Count header level
                level = len(re.match(r'^#+', part).group())
                title = part.strip('# ').strip()
                
                current_section = {
                    "title": title,
                    "content": "",
                    "level": level
                }
            else:
                # This is content
                current_section["content"] += part
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean up the text
        text = text.strip()
        
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Find the end of this chunk
            end = min(start + self.chunk_size, text_len)
            
            # If we're not at the end, try to break at a sentence or paragraph
            if end < text_len:
                # Look for paragraph break
                break_pos = text.rfind('\n\n', start, end)
                if break_pos == -1:
                    # Look for sentence end
                    break_pos = text.rfind('. ', start, end)
                if break_pos == -1:
                    # Look for any space
                    break_pos = text.rfind(' ', start, end)
                if break_pos != -1 and break_pos > start:
                    end = break_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap, ensuring we make progress
            start = max(start + 1, end - self.chunk_overlap)
            
            # Safety check to prevent infinite loop
            if start >= text_len:
                break
        
        return chunks
    
    def process_document(
        self,
        file_path: str,
        chunk_by_section: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a markdown document into chunks with metadata.
        
        Args:
            file_path: Path to the markdown file
            chunk_by_section: If True, respect section boundaries
            
        Returns:
            List of chunks with metadata
        """
        print(f"Loading document from: {file_path}")
        content = self.load_markdown(file_path)
        print(f"Document loaded: {len(content)} characters")
        
        if chunk_by_section:
            print("Extracting sections...")
            sections = self.extract_sections(content)
            print(f"Found {len(sections)} sections")
            
            chunks = []
            
            for idx, section in enumerate(sections):
                print(f"Processing section {idx+1}/{len(sections)}: {section['title'][:50]}...")
                
                # Skip empty sections
                if not section["content"].strip():
                    continue
                
                try:
                    section_chunks = self.chunk_text(section["content"])
                    print(f"  Created {len(section_chunks)} chunks")
                    
                    for chunk_idx, chunk in enumerate(section_chunks):
                        chunks.append({
                            "text": chunk,
                            "metadata": {
                                "section_title": section["title"],
                                "section_level": section["level"],
                                "chunk_index": chunk_idx,
                                "total_chunks": len(section_chunks),
                                "source": file_path
                            }
                        })
                except Exception as e:
                    print(f"  Warning: Error processing section '{section['title']}': {e}")
                    continue
        else:
            # Simple chunking without section awareness
            print("Chunking document without section awareness...")
            text_chunks = self.chunk_text(content)
            chunks = [
                {
                    "text": chunk,
                    "metadata": {
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks),
                        "source": file_path
                    }
                }
                for idx, chunk in enumerate(text_chunks)
            ]
        
        print(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and special characters.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
