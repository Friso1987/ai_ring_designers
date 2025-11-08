# design/scad_generator.py

from design.parameters import DesignParameters
from typing import Optional

class SCADGenerator:
    """Generates OpenSCAD code from design parameters"""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
    
    def generate(self, params: DesignParameters) -> str:
        """Generate complete OpenSCAD code"""
        
        code = f"""
// AI-Generated Design
// Generation timestamp: {{timestamp}}
// Score: {{score}}

// Parameters
inner_diameter = {params.inner_diameter};
wall_thickness = {params.wall_thickness};
ring_height = {params.ring_height};
gap_width = {params.gap_width};
lug_width = {params.lug_width};
lug_height = {params.lug_height};
lug_thickness = {params.lug_thickness};
screw_hole_diameter = 3.5;
insert_hole_diameter = 4.5;

// Organic growth parameters
num_main_veins = {params.num_main_veins};
height_layers = {params.height_layers};
lateral_wander_base = {params.lateral_wander};
radial_wander_base = {params.radial_wander};
thickness_variation = {params.organic_variation};
branch_probability = {params.branch_probability};
sub_branch_length = {params.sub_branch_length};
node_density = {params.node_density};
inner_clearance = 1;

// Berekeningen
outer_diameter = inner_diameter + (2 * wall_thickness);
radius = outer_diameter / 2;
inner_radius = inner_diameter / 2;
branch_start_radius = inner_radius + inner_clearance;

// Random functies
function pseudo_random(seed, x) = 
    abs(sin(seed * 12.9898 + x * 78.233) * 43758.5453) - floor(abs(sin(seed * 12.9898 + x * 78.233) * 43758.5453));

function smooth_noise(seed, t, scale) = 
    sin(t * scale + seed) * 0.5 + 
    sin(t * scale * 2.13 + seed * 1.7) * 0.25 +
    sin(t * scale * 4.37 + seed * 2.3) * 0.125;

function layer_lateral_wander(seed, layer_index) = 
    lateral_wander_base * (0.5 + pseudo_random(seed, layer_index) * 1.5);

function layer_radial_wander(seed, layer_index) = 
    radial_wander_base * (0.3 + pseudo_random(seed + 1000, layer_index) * 1.4);

function layer_twist(seed, layer_index) = 
    (pseudo_random(seed + 2000, layer_index) - 0.5) * 40 * {params.layer_twist_factor};

function layer_branch_bias(seed, layer_index) =
    (pseudo_random(seed + 3000, layer_index) - 0.5) * 2 * {params.layer_branch_bias};

module organic_node(r, h, seed) {{
    scale([
        1 + smooth_noise(seed, r, 0.5) * 0.2,
        1 + smooth_noise(seed + 10, r, 0.7) * 0.2,
        1
    ])
    cylinder(h=h, r=r, $fn=16);
}}

module organic_segment(p1, p2, r1, r2, h1, h2, seed) {{
    hull() {{
        translate([p1[0], p1[1], h1])
            cylinder(h=0.1, r=r1, $fn=16);
        translate([p2[0], p2[1], h2])
            cylinder(h=0.1, r=r2, $fn=16);
    }}
}}

module multi_layer_vein(start_angle, seed, is_main = true) {{
    base_thickness = is_main ? {params.vein_thickness} : {params.vein_thickness * 0.7};
    
    for (layer = [0:height_layers-1]) {{
        h_start = layer * (ring_height / height_layers);
        h_end = (layer + 1) * (ring_height / height_layers);
        
        layer_lateral = layer_lateral_wander(seed, layer);
        layer_radial = layer_radial_wander(seed, layer);
        layer_rotation = layer_twist(seed, layer);
        layer_branch_direction = layer_branch_bias(seed, layer);
        
        segments_in_layer = 3 + floor(pseudo_random(seed + 500, layer) * 4);
        
        for (seg = [0:segments_in_layer-1]) {{
            t_layer = seg / segments_in_layer;
            next_t_layer = (seg + 1) / segments_in_layer;
            
            total_t = (layer + t_layer) / height_layers;
            next_total_t = (layer + next_t_layer) / height_layers;
            
            base_angle_offset = smooth_noise(seed, layer + seg, 0.6) * layer_lateral;
            next_base_angle_offset = smooth_noise(seed, layer + seg + 1, 0.6) * layer_lateral;
            
            angle1 = start_angle + base_angle_offset + layer_rotation * t_layer;
            angle2 = start_angle + next_base_angle_offset + layer_rotation * next_t_layer;
            
            radial_noise1 = smooth_noise(seed * 1.3, layer * 10 + seg, 0.9) * layer_radial;
            radial_noise2 = smooth_noise(seed * 1.3, layer * 10 + seg + 1, 0.9) * layer_radial;
            
            r1 = branch_start_radius + (radius - branch_start_radius - 1) * total_t + radial_noise1;
            r2 = branch_start_radius + (radius - branch_start_radius - 1) * next_total_t + radial_noise2;
            
            thickness_noise = 1 - (smooth_noise(seed * 0.7, layer + seg, 0.5) * thickness_variation);
            thickness1 = base_thickness * thickness_noise * (1 - total_t * 0.15);
            thickness2 = base_thickness * thickness_noise * (1 - next_total_t * 0.15);
            
            p1 = [r1 * cos(angle1), r1 * sin(angle1)];
            p2 = [r2 * cos(angle2), r2 * sin(angle2)];
            
            z1 = h_start + t_layer * (h_end - h_start);
            z2 = h_start + next_t_layer * (h_end - h_start);
            
            organic_segment(p1, p2, thickness1, thickness2, z1, z2, seed + layer * 100 + seg);
            
            random_node = pseudo_random(seed + 4000, layer * 10 + seg);
            if (random_node < node_density * 0.4) {{
                translate([p1[0], p1[1], z1])
                    cylinder(h=0.5, r=thickness1 * 1.5, $fn=16);
            }}
            
            branch_random = pseudo_random(seed + 5000, layer * 10 + seg);
            if (is_main && branch_random < branch_probability && seg > 0) {{
                branch_seed = seed * 13 + layer * 97 + seg * 7;
                branch_angle_base = angle1 + (pseudo_random(branch_seed, 0) - 0.5) * {params.branch_angle_variation};
                branch_angle = branch_angle_base + layer_branch_direction * 15;
                
                for (j = [0:2]) {{
                    bj_t = j / 3;
                    next_bj_t = (j + 1) / 3;
                    
                    lateral_curve = smooth_noise(branch_seed, j, 1.2) * 25;
                    next_lateral_curve = smooth_noise(branch_seed, j + 1, 1.2) * 25;
                    
                    ba1 = branch_angle + lateral_curve;
                    ba2 = branch_angle + next_lateral_curve;
                    
                    br1 = r1 + bj_t * 5 + smooth_noise(branch_seed, j, 0.7) * 2;
                    br2 = r1 + next_bj_t * 5 + smooth_noise(branch_seed, j + 1, 0.7) * 2;
                    
                    bz1 = z1 + (pseudo_random(branch_seed, j) - 0.5) * 3;
                    bz2 = z1 + (pseudo_random(branch_seed, j + 1) - 0.5) * 3;
                    
                    bt1 = thickness1 * 0.5 * (1 - bj_t * 0.4);
                    bt2 = thickness1 * 0.5 * (1 - next_bj_t * 0.4);
                    
                    bp1 = [br1 * cos(ba1), br1 * sin(ba1)];
                    bp2 = [br2 * cos(ba2), br2 * sin(ba2)];
                    
                    organic_segment(bp1, bp2, bt1, bt2, bz1, bz2, branch_seed + j);
                }}
            }}
        }}
    }}
}}

module cross_layer_connections() {{
    for (i = [0:num_main_veins-1]) {{
        seed_base = 7000 + i * 50;
        angle1 = i * 360 / num_main_veins;
        angle2 = ((i + 1) % num_main_veins) * 360 / num_main_veins;
        
        for (conn = [0:3]) {{
            connection_t = 0.25 + conn * 0.2;
            conn_radius = branch_start_radius + (radius - branch_start_radius) * connection_t;
            
            z_start = pseudo_random(seed_base, conn) * ring_height * 0.6;
            z_end = z_start + (pseudo_random(seed_base + 100, conn) * 2 + 1) * (ring_height * 0.2);
            
            arc_steps = 5;
            for (j = [0:arc_steps-1]) {{
                arc_t = j / arc_steps;
                next_arc_t = (j + 1) / arc_steps;
                
                a1 = angle1 + (angle2 - angle1) * arc_t;
                a2 = angle1 + (angle2 - angle1) * next_arc_t;
                
                bulge = smooth_noise(seed_base, j + conn, 0.8) * 2.5;
                next_bulge = smooth_noise(seed_base, j + 1 + conn, 0.8) * 2.5;
                
                r1 = conn_radius + bulge;
                r2 = conn_radius + next_bulge;
                
                z1 = z_start + arc_t * (z_end - z_start);
                z2 = z_start + next_arc_t * (z_end - z_start);
                
                thick = 1.3 * (1 - abs(arc_t - 0.5) * 0.5);
                next_thick = 1.3 * (1 - abs(next_arc_t - 0.5) * 0.5);
                
                p1 = [r1 * cos(a1), r1 * sin(a1)];
                p2 = [r2 * cos(a2), r2 * sin(a2)];
                
                organic_segment(p1, p2, thick, next_thick, z1, z2, seed_base + j + conn * 100);
            }}
        }}
    }}
}}

module full_organic_structure() {{
    union() {{
        difference() {{
            cylinder(h=ring_height, r=branch_start_radius + 0.3, $fn=100);
            translate([0, 0, -0.5])
                cylinder(h=ring_height+1, r=inner_radius, $fn=100);
        }}
        
        for (i = [0:num_main_veins-1]) {{
            base_angle = i * 360 / num_main_veins;
            vein_seed = 42 + i * 137;
            multi_layer_vein(base_angle, vein_seed, true);
        }}
        
        cross_layer_connections();
        
        translate([0, 0, ring_height - 0.5])
        difference() {{
            cylinder(h=0.5, r=branch_start_radius + 0.5, $fn=100);
            translate([0, 0, -0.5])
                cylinder(h=2, r=inner_radius, $fn=100);
        }}
    }}
}}

// Main structure
difference() {{
    union() {{
        full_organic_structure();
        
        translate([-gap_width/2 - lug_thickness, radius - lug_width/2, 0]) {{
            hull() {{
                cube([lug_thickness, lug_width, 2]);
                translate([lug_thickness/2, lug_width/2, lug_height - 2])
                    sphere(r=4, $fn=30);
                translate([lug_thickness/2, lug_width/2, lug_height])
                    sphere(r=3, $fn=30);
            }}
        }}
        
        translate([gap_width/2, radius - lug_width/2, 0]) {{
            hull() {{
                cube([lug_thickness, lug_width, 2]);
                translate([lug_thickness/2, lug_width/2, lug_height - 2])
                    sphere(r=4, $fn=30);
                translate([lug_thickness/2, lug_width/2, lug_height])
                    sphere(r=3, $fn=30);
            }}
        }}
    }}
    
    translate([0, 0, -1])
        cylinder(h=ring_height + 2, r=inner_radius, $fn=100);
    
    translate([-gap_width/2, -1, -0.5])
        cube([gap_width, radius + 10, ring_height + 1]);
    
    translate([-gap_width/2 - lug_thickness/2, radius, lug_height * 0.6])
        rotate([0, 90, 0])
            cylinder(h=lug_thickness+2, r=insert_hole_diameter/2, $fn=30);
    
    translate([gap_width/2 - 1, radius, lug_height * 0.6])
        rotate([0, 90, 0])
            cylinder(h=lug_thickness+2, r=screw_hole_diameter/2, $fn=30);
}}
"""
        return code
