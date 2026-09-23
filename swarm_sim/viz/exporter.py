from typing import List, Optional
import os
from PIL import Image

class GIFExporter:
    """Pillow GIF exporter assembling rendered simulation frames into an animated GIF."""
    def __init__(self):
        self.frames: List[Image.Image] = []

    def add_frame(self, frame: Image.Image) -> None:
        """Append a Pillow Image frame."""
        self.frames.append(frame.copy())

    def save_gif(self, output_path: str, fps: int = 10, loop: int = 0) -> None:
        """Export accumulated frames to an animated GIF."""
        if not self.frames:
            raise ValueError("No frames added to GIFExporter.")
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        duration_ms = int(1000.0 / fps)
        
        self.frames[0].save(
            output_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration_ms,
            loop=loop
        )
