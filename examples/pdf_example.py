"""Example usage of the PDF processor library."""

import sys
import os

# Get the absolute path to the library directory
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(os.path.dirname(current_dir), 'library')
sys.path.append(library_dir)

from pdf_processor import PDFProcessor, TextSplitterConfig, PDFProcessingError

def main():
    # Create a processor with custom configuration
    config = TextSplitterConfig(
        chunk_size=800,
        chunk_overlap=100,
        separator=' '
    )
    processor = PDFProcessor(config)
    
    try:
        # Load and process a PDF
        chunks = processor.process_pdf("PDF-docs/CFDs - General.pdf")
        print(f"Processed PDF into {len(chunks)} chunks")
        
        # Print first chunk content
        if chunks:
            print("\nFirst chunk content:")
            print(chunks[0].page_content[:200])
        
        # Try different splitting configuration
        new_config = TextSplitterConfig(chunk_size=500, chunk_overlap=50)
        processor.update_config(new_config)
        
        chunks = processor.process_pdf("PDF-docs/KYC_Data.pdf", use_recursive=False)
        print(f"\nProcessed PDF with new config into {len(chunks)} chunks")
        
    except PDFProcessingError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
