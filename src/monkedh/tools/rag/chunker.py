from typing import List, Dict, Any
import re
import math


class DocumentChunker:
    """Handles document loading and chunking."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""] # Separators used for recursive splitting
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to use for recursive splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def load_markdown(self, file_path: str) -> str:
        """
        Load file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and special characters.
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def recursive_split_text(self, text: str) -> List[str]:
        """
        Recursively splits text using the list of separators until chunks fit size limit.
        """
        if not text:
            return []
            
        # Base case: If the chunk is small enough, no further splitting is needed
        if len(text) <= self.chunk_size:
            return [text]

        # Find the best separator to use
        best_separator = None
        for sep in self.separators:
            if sep in text:
                best_separator = sep
                break
        
        # If no separator found, use fixed-size chunking (character split)
        if best_separator is None:
            best_separator = "" # Fallback to character-level splitting

        splits = text.split(best_separator)
        final_chunks = []
        current_chunk = ""

        for split in splits:
            # Check if adding the current split exceeds the max size
            if len(current_chunk) + len(best_separator) + len(split) > self.chunk_size and current_chunk:
                # The current chunk is ready, but it must include overlap from the previous one.
                # Since we are iterating through splits, we just append the current_chunk 
                # and recurse on the split itself if it is still too large.
                
                # If the split is too large, recurse on it first
                if len(split) > self.chunk_size:
                    final_chunks.extend(self.recursive_split_text(split))
                    continue

                # Add the accumulated chunk (before the new split)
                final_chunks.append(current_chunk)

                # Set up the next chunk with overlap
                overlap_text = current_chunk[-(self.chunk_overlap):]
                current_chunk = overlap_text + best_separator + split
            else:
                # Accumulate the split into the current chunk
                current_chunk += (best_separator + split if current_chunk else split)

        # Add the last accumulated chunk
        if current_chunk:
             final_chunks.append(current_chunk)
        
        # Final pass for fixed-size/overlap in case of no good separators
        if best_separator == "" and final_chunks and len(final_chunks[0]) > self.chunk_size:
            text_to_chunk = final_chunks[0]
            start = 0
            chunks = []
            while start < len(text_to_chunk):
                end = start + self.chunk_size
                chunks.append(text_to_chunk[start:end].strip())
                start += (self.chunk_size - self.chunk_overlap)
            return chunks

        return [chunk.strip() for chunk in final_chunks if chunk.strip()]


    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive splitting with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks, respecting chunk_size and chunk_overlap
        """
        # The main entry point calls the recursive splitter
        return self.recursive_split_text(text)

    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract Markdown sections from content (based on # headings).
        (Implementation for the missing method)
        """
        sections = []
        heading_pattern = re.compile(r'(^#{1,6}\s+.*)', re.MULTILINE)
        splits = re.split(heading_pattern, content)

        current_content = splits[0].strip() if splits else ""

        if current_content:
             sections.append({
                "title": "Document Start (No Header)",
                "content": current_content,
                "level": 0,
                "source_type": "markdown"
            })

        for i in range(1, len(splits), 2):
            if i + 1 < len(splits):
                heading = splits[i].strip()
                content = splits[i+1].strip()

                match = re.match(r'(#+)\s+(.*)', heading)
                level = len(match.group(1)) if match else 1
                title = match.group(2).strip() if match else heading

                sections.append({
                    "title": title,
                    "content": content,
                    "level": level,
                    "source_type": "markdown"
                })

        return sections
    
    def extract_rt_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract RT sections from the first aid manual text format.
        (Original method, with added metadata)
        """
        sections = []
        
        # Split by section separators (handle file start)
        parts = re.split(r'(?:^|\n)={50,}(?:\n|$)', content)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            lines = part.split('\n')
            if not lines:
                continue
            
            first_line = lines[0].strip()
            
            # Handle non-RT intro section
            if not first_line.startswith('RT') and i == 0:
                 sections.append({
                    "title": "Document Introduction",
                    "rt_id": "N/A",
                    "numero_rt": 0,
                    "content": part,
                    "subsections": {},
                    "level": 0,
                    "source_type": "rt_manual"
                })
                
            
            rt_match = re.match(r'RT(\d+):\s*(.+)', first_line)
            if not rt_match:
                continue
            
            rt_number = int(rt_match.group(1))
            rt_title = rt_match.group(2).strip()
            
            content_lines = lines[1:]
            
            # Parse subsections
            subsections = {}
            current_subsection = None
            current_content = []
            
            for line in content_lines:
                line = line.strip()
                if not line:
                    continue
                
                if re.match(r'^[A-Z\s]+:$', line) and len(line) > 3:
                    if current_subsection and current_content:
                        subsections[current_subsection] = '\n'.join(current_content).strip()
                    
                    current_subsection = line[:-1].lower().replace(' ', '_')
                    current_content = []
                else:
                    if current_subsection:
                        current_content.append(line)
            
            if current_subsection and current_content:
                subsections[current_subsection] = '\n'.join(current_content).strip()
            
            sections.append({
                "title": rt_title,
                "rt_id": f"RT{rt_number}",
                "numero_rt": rt_number,
                "content": part,
                "subsections": subsections,
                "level": 1,
                "source_type": "rt_manual"
            })
        return sections

    def process_document(
        self,
        file_path: str,
        chunk_by_section: bool = True,
        format: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Process a document into chunks with metadata.
        """
        content = self.load_markdown(file_path)
        # Improvement: Clean text immediately after loading
        content = self.clean_text(content)
        
        if format == "auto":
            if "==================================================" in content and "RT" in content:
                format = "rt_manual"
            else:
                format = "markdown"

        if chunk_by_section:
            if format == "rt_manual":
                sections = self.extract_rt_sections(content)
            else:
                # Fixed: Use the implemented markdown extractor
                sections = self.extract_sections(content)
            
            chunks = []
            
            for section in sections:
                if not section.get("content") or not section["content"].strip():
                    continue
                
                try:
                    # Fixed: Use the new recursive chunker which respects size and overlap
                    section_chunks = self.chunk_text(section["content"])
                    
                    for chunk_idx, chunk in enumerate(section_chunks):
                        metadata = {
                            "section_title": section["title"],
                            "chunk_index": chunk_idx,
                            "total_chunks": len(section_chunks),
                            "source": file_path,
                            "level": section.get("level", 1)
                        }
                        
                        if "rt_id" in section and section["source_type"] == "rt_manual":
                            metadata.update({
                                "rt_id": section["rt_id"],
                                "numero_rt": section["numero_rt"]
                            })
                        
                        chunks.append({
                            "text": chunk,
                            "metadata": metadata
                        })
                except Exception as e:
                    # In a production setting, use proper logging
                    print(f"  Warning: Error processing section '{section['title']}': {e}")
                    continue
        else:
            # Simple chunking without section awareness
            text_chunks = self.chunk_text(content)
            chunks = [
                {
                    "text": chunk,
                    "metadata": {
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks),
                        "source": file_path,
                        "level": 0
                    }
                }
                for idx, chunk in enumerate(text_chunks)
            ]
        
        return chunks