# design/renderer.py

import subprocess
import tempfile
from pathlib import Path
import trimesh
from PIL import Image
import numpy as np
from typing import Optional
import logging

class DesignRenderer:
    """Renders OpenSCAD code to mesh and images"""
    
    def __init__(self, openscad_path: str = "openscad"):
        self.openscad_path = openscad_path
        self.logger = logging.getLogger(__name__)
        
        # Test if OpenSCAD is available
        try:
            subprocess.run(
                [self.openscad_path, "--version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            self.logger.info(f"OpenSCAD found at: {self.openscad_path}")
        except Exception as e:
            self.logger.warning(f"OpenSCAD not found: {e}")
            self.logger.warning("Rendering will use fallback methods")
    
    def render_to_mesh(
        self, 
        scad_code: str, 
        timeout: int = 60
    ) -> Optional[trimesh.Trimesh]:
        """Render OpenSCAD code to 3D mesh"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write SCAD file
            scad_file = tmpdir / "design.scad"
            with open(scad_file, 'w') as f:
                f.write(scad_code)
            
            # Output STL file
            stl_file = tmpdir / "design.stl"
            
            try:
                # Run OpenSCAD
                cmd = [
                    self.openscad_path,
                    "-o", str(stl_file),
                    str(scad_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                
                if result.returncode != 0:
                    self.logger.error(f"OpenSCAD error: {result.stderr}")
                    return None
                
                # Load mesh
                if stl_file.exists():
                    mesh = trimesh.load(str(stl_file))
                    return mesh
                else:
                    self.logger.error("STL file not generated")
                    return None
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"Rendering timeout after {timeout}s")
                return None
            except Exception as e:
                self.logger.error(f"Rendering error: {e}")
                return None
    
    def render_to_image(
        self, 
        scad_code: str,
        size: tuple = (800, 600),
        camera_angle: tuple = (60, 0, 45),
        timeout: int = 60
    ) -> Optional[Image.Image]:
        """Render OpenSCAD code to PNG image"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write SCAD file
            scad_file = tmpdir / "design.scad"
            with open(scad_file, 'w') as f:
                f.write(scad_code)
            
            # Output PNG file
            png_file = tmpdir / "design.png"
            
            try:
                # Run OpenSCAD with camera settings
                cmd = [
                    self.openscad_path,
                    "-o", str(png_file),
                    "--imgsize", f"{size[0]},{size[1]}",
                    "--camera", f"{camera_angle[0]},{camera_angle[1]},{camera_angle[2]},0,0,0",
                    "--viewall",
                    "--autocenter",
                    str(scad_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                
                if result.returncode != 0:
                    self.logger.error(f"OpenSCAD error: {result.stderr}")
                    return None
                
                # Load image
                if png_file.exists():
                    image = Image.open(png_file)
                    return image.copy()  # Copy before temp dir is deleted
                else:
                    self.logger.error("PNG file not generated")
                    return None
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"Rendering timeout after {timeout}s")
                return None
            except Exception as e:
                self.logger.error(f"Rendering error: {e}")
                return None
    
    def render_multi_angle(
        self,
        scad_code: str,
        angles: list = None
    ) -> list:
        """Render from multiple angles"""
        if angles is None:
            angles = [
                (60, 0, 45),    # Front-right
                (60, 0, 135),   # Back-right
                (60, 0, 225),   # Back-left
                (60, 0, 315),   # Front-left
                (0, 0, 0),      # Top
            ]
        
        images = []
        for angle in angles:
            img = self.render_to_image(scad_code, camera_angle=angle)
            if img:
                images.append(img)
        
        return images