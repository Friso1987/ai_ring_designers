# design/mock_renderer.py

"""Mock renderer voor testing zonder OpenSCAD"""

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

class MockRenderer:
    """Fake renderer die dummy data teruggeeft"""
    
    def __init__(self):
        pass
    
    def render_to_mesh(self, scad_code: str) -> trimesh.Trimesh:
        """Create a simple dummy mesh"""
        # Maak een simpele cilinder als placeholder
        # Dit werkt beter dan torus voor compatibiliteit
        try:
            mesh = trimesh.creation.cylinder(
                radius=27.5,  # (55mm diameter / 2)
                height=20,
                sections=32
            )
        except:
            # Fallback: maak een basic box als mesh
            mesh = trimesh.creation.box(extents=[55, 55, 20])
        
        return mesh
    
    def render_to_image(self, scad_code: str, **kwargs) -> Image.Image:
        """Create a dummy image with some visual feedback"""
        # Maak een grijze placeholder image
        img = Image.new('RGB', (800, 600), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        
        # Teken een cirkel als representatie
        center_x, center_y = 400, 300
        radius = 150
        
        # Buitenste ring
        draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            outline=(100, 100, 100),
            width=5
        )
        
        # Binnenste ring
        inner_radius = 100
        draw.ellipse(
            [center_x - inner_radius, center_y - inner_radius,
             center_x + inner_radius, center_y + inner_radius],
            outline=(150, 150, 150),
            width=3
        )
        
        # Voeg wat "organic" lijnen toe
        num_lines = np.random.randint(6, 12)
        for i in range(num_lines):
            angle = i * 360 / num_lines
            angle_rad = np.radians(angle)
            
            x1 = center_x + inner_radius * np.cos(angle_rad)
            y1 = center_y + inner_radius * np.sin(angle_rad)
            x2 = center_x + radius * np.cos(angle_rad)
            y2 = center_y + radius * np.sin(angle_rad)
            
            # Voeg wat variatie toe
            wobble = np.random.randint(-10, 10)
            x2 += wobble
            y2 += wobble
            
            draw.line([(x1, y1), (x2, y2)], fill=(120, 120, 120), width=2)
        
        # Tekst
        try:
            draw.text((50, 50), "MOCK RENDER", fill=(180, 180, 180))
            draw.text((50, 550), f"Generated: {np.random.randint(1000, 9999)}", fill=(180, 180, 180))
        except:
            pass  # Font not available
        
        return img
    
    def render_multi_angle(self, scad_code: str, angles: list = None) -> list:
        """Render from multiple angles (all return same mock image)"""
        if angles is None:
            angles = [(60, 0, 45), (60, 0, 135), (60, 0, 225), (60, 0, 315), (0, 0, 0)]
        
        images = []
        for _ in angles:
            img = self.render_to_image(scad_code)
            images.append(img)
        
        return images


class FastRenderer:
    """Faster renderer met lagere resolutie"""
    
    def __init__(self, openscad_path: str = "openscad"):
        from design.renderer import DesignRenderer
        self.renderer = DesignRenderer(openscad_path)
    
    def render_to_mesh(self, scad_code: str) -> trimesh.Trimesh:
        """Render with increased timeout"""
        return self.renderer.render_to_mesh(scad_code, timeout=120)
    
    def render_to_image(self, scad_code: str, **kwargs) -> Image.Image:
        """Render with lower resolution"""
        return self.renderer.render_to_image(
            scad_code, 
            size=(400, 300),
            **kwargs
        )