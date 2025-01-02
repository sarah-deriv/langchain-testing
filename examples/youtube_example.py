"""Example usage of the YouTube processor library."""

import sys
import os

# Get the absolute path to the library directory
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(os.path.dirname(current_dir), 'library')
sys.path.append(library_dir)

from youtube_processor import YouTubeProcessor, TextSplitterConfig, YouTubeProcessingError

def main():
    # Create a processor with custom configuration
    config = TextSplitterConfig(
        chunk_size=800,
        chunk_overlap=100,
        separator=' '
    )
    processor = YouTubeProcessor(config)
    
    try:
        # Example YouTube video URL
        video_url = "https://www.youtube.com/watch?v=pJY0mBWHPw4"
        
        # Load and process the video transcript
        chunks = processor.process_video(video_url)
        print(f"Processed video into {len(chunks)} chunks")
        
        # Print first chunk content
        if chunks:
            print("\nFirst chunk content:")
            print(chunks[0].page_content[:200])
        
        # Try different splitting configuration
        new_config = TextSplitterConfig(chunk_size=500, chunk_overlap=50)
        processor.update_config(new_config)
        
        chunks = processor.process_video(video_url, use_recursive=False)
        print(f"\nProcessed video with new config into {len(chunks)} chunks")
        
    except YouTubeProcessingError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
