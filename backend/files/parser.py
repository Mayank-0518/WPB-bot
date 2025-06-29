"""
File Parser for WhatsApp AI Second Brain Assistant
Handles PDF, image OCR, and text extraction from various file formats

Example Usage:
INPUT: PDF file or image with text
OUTPUT: Extracted text content ready for processing

# PDF Processing
text = await parser.parse_pdf("document.pdf")

# Image OCR  
text = await parser.parse_image("screenshot.jpg")

# URL Content
text = await parser.parse_url("https://example.com/article")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import aiofiles
import httpx
from PIL import Image
import fitz  # PyMuPDF
from backend.utils.config import config

logger = logging.getLogger(__name__)

class FileParser:
    """Parse various file formats and extract text content"""
    
    def __init__(self):
        self.max_file_size = config.MAX_FILE_SIZE
        self.upload_dir = config.UPLOAD_DIR
        
        # Supported file types
        self.supported_types = {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            'text': ['.txt', '.md', '.csv', '.json'],
            'document': ['.docx', '.doc', '.rtf']  # Limited support
        }
    
    async def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse any supported file and extract text
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Dict with extracted content, metadata, and file info
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            # Determine file type
            file_type = self._get_file_type(file_path)
            
            # Parse based on type
            if file_type == 'pdf':
                content = await self.parse_pdf(str(file_path))
            elif file_type == 'image':
                content = await self.parse_image(str(file_path))
            elif file_type == 'text':
                content = await self.parse_text_file(str(file_path))
            elif file_type == 'document':
                content = await self.parse_document(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Prepare result
            result = {
                "content": content,
                "file_type": file_type,
                "file_name": file_path.name,
                "file_size": file_size,
                "word_count": len(content.split()) if content else 0,
                "metadata": {
                    "source": "file_upload",
                    "file_extension": file_path.suffix.lower(),
                    "parsing_method": file_type
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"File parsing failed for {file_path}: {e}")
            raise
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension"""
        ext = file_path.suffix.lower()
        
        for file_type, extensions in self.supported_types.items():
            if ext in extensions:
                return file_type
        
        return 'unknown'
    
    async def parse_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF files using PyMuPDF
        
        Example:
        INPUT: "report.pdf" (multi-page business report)
        OUTPUT: "Executive Summary\nThis report covers Q2 performance metrics..."
        """
        try:
            doc = fitz.open(file_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    text_content.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            
            # Clean up the text
            full_text = self._clean_text(full_text)
            
            logger.info(f"Extracted {len(full_text)} characters from PDF: {file_path}")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF parsing failed for {file_path}: {e}")
            raise Exception(f"PDF parsing failed: {str(e)}")
    
    async def parse_image(self, file_path: str) -> str:
        """
        Extract text from images using OCR
        
        Example:
        INPUT: "screenshot.png" (image containing text)
        OUTPUT: "Meeting Agenda\n1. Project Updates\n2. Budget Review..."
        """
        try:
            # For now, return a placeholder since OCR requires additional setup
            # In production, you would use pytesseract or similar
            logger.warning(f"OCR not fully implemented. Image file: {file_path}")
            
            # Basic image info
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    
                return f"[Image file processed: {os.path.basename(file_path)}]\nDimensions: {width}x{height}, Mode: {mode}\nNote: OCR text extraction requires additional setup."
                
            except Exception as img_error:
                return f"[Image file detected: {os.path.basename(file_path)}]\nError processing image: {str(img_error)}"
            
        except Exception as e:
            logger.error(f"Image parsing failed for {file_path}: {e}")
            return f"Error reading image: {str(e)}"
    
    async def parse_text_file(self, file_path: str) -> str:
        """
        Read plain text files
        
        Example:
        INPUT: "notes.txt"
        OUTPUT: Content of the text file
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
                
            # Clean and limit content
            content = self._clean_text(content)
            
            logger.info(f"Read {len(content)} characters from text file: {file_path}")
            return content
            
        except UnicodeDecodeError:
            # Try different encodings
            try:
                async with aiofiles.open(file_path, 'r', encoding='latin-1') as file:
                    content = await file.read()
                return self._clean_text(content)
            except Exception:
                return f"Error: Could not decode text file {file_path}"
                
        except Exception as e:
            logger.error(f"Text file parsing failed for {file_path}: {e}")
            return f"Error reading text file: {str(e)}"
    
    async def parse_document(self, file_path: str) -> str:
        """
        Parse document files (limited support)
        
        Note: Full DOCX support would require python-docx library
        """
        logger.warning(f"Document parsing not fully implemented for: {file_path}")
        return f"[Document file detected: {os.path.basename(file_path)}]\nNote: Full document parsing requires additional libraries."
    
    async def parse_url(self, url: str) -> str:
        """
        Extract content from web URLs
        
        Example:
        INPUT: "https://example.com/article"
        OUTPUT: "Article Title\nArticle content extracted from webpage..."
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Basic HTML text extraction (simplified)
                content = response.text
                
                # Remove HTML tags (basic approach)
                import re
                text = re.sub(r'<[^>]+>', '', content)
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                
                # Clean and limit
                text = self._clean_text(text)
                
                logger.info(f"Extracted {len(text)} characters from URL: {url}")
                return text[:5000]  # Limit URL content
                
        except Exception as e:
            logger.error(f"URL parsing failed for {url}: {e}")
            return f"Error fetching content from URL: {str(e)}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Limit length
        if len(text) > 50000:  # 50K character limit
            text = text[:50000] + "\n\n[Content truncated...]"
        
        return text.strip()
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return path"""
        try:
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename)
            file_path = self.upload_dir / safe_filename
            
            # Ensure upload directory exists
            self.upload_dir.mkdir(exist_ok=True)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Saved uploaded file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file {filename}: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        import re
        import uuid
        
        # Remove dangerous characters
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Add UUID prefix to avoid conflicts
        name_parts = safe_name.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            return f"{uuid.uuid4().hex[:8]}_{name}.{ext}"
        else:
            return f"{uuid.uuid4().hex[:8]}_{safe_name}"
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information without parsing content"""
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                "name": path.name,
                "size": stat.st_size,
                "type": self._get_file_type(path),
                "extension": path.suffix.lower(),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "exists": path.exists()
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"error": str(e)}
    
    def is_supported(self, filename: str) -> bool:
        """Check if file type is supported"""
        ext = Path(filename).suffix.lower()
        return any(ext in extensions for extensions in self.supported_types.values())

# Global parser instance
file_parser = FileParser()

# Test function
async def test_file_parser():
    """Test file parsing functionality"""
    
    print("🧪 Testing File Parser")
    print("=" * 50)
    
    # Test URL parsing
    try:
        url_content = await file_parser.parse_url("https://httpbin.org/html")
        print(f"✅ URL parsing: {len(url_content)} characters extracted")
        print(f"   Preview: {url_content[:100]}...")
    except Exception as e:
        print(f"❌ URL parsing failed: {e}")
    
    # Test file type detection
    test_files = ["test.pdf", "image.jpg", "notes.txt", "doc.docx"]
    for filename in test_files:
        supported = file_parser.is_supported(filename)
        file_type = file_parser._get_file_type(Path(filename))
        print(f"✅ {filename}: {file_type} ({'supported' if supported else 'not supported'})")
    
    print("✅ File parser tests completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_file_parser())
