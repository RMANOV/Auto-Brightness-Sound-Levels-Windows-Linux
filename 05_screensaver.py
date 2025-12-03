#!/usr/bin/env python3
"""
Lubuntu-style Multi-Screensaver
11 Mandala & 3D Visualizations with random rotation every 7 minutes
Pure Tkinter - zero dependencies
"""

import math
import random
import tkinter as tk
from datetime import datetime

# === CONFIG ===
CONFIG = {
    'fps': 30,
    'switch_interval': 7 * 60 * 1000,  # 7 minutes in ms
    'rotation_speed': 0.008,
    'color_cycle_speed': 0.004,
    'num_stars': 40,
    'star_speed': 0.3,
    'star_size': 3,
    'bg_color': '#06080c',
    'line_width': 3,
    'num_trail_points': 500,
    'clock_height': 80,
    'clock_font_size': 48,
}


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """Convert HSV to hex color"""
    try:
        h = h % 1.0
        i = int(h * 6)
        f = (h * 6) - i
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
        return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
    except Exception:
        return '#8080ff'


# =============================================================================
# VISUALIZATION 1: Tesseract (4D Hypercube)
# =============================================================================
class Tesseract:
    """4D Hypercube projection"""
    name = "4D Tesseract"
    num_lines = 32

    def __init__(self):
        self.angle_xy = self.angle_zw = self.angle_xz = self.angle_yz = 0.0
        self.hue_offset = 0.0
        self.vertices_4d = [[x, y, z, w] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]]
        self.edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                      if sum(1 for k in range(4) if self.vertices_4d[i][k] != self.vertices_4d[j][k]) == 1]

    def update(self):
        self.angle_xy += CONFIG['rotation_speed'] * 1.5
        self.angle_zw += CONFIG['rotation_speed'] * 1.05
        self.angle_xz += CONFIG['rotation_speed']
        self.angle_yz += CONFIG['rotation_speed'] * 0.5
        self.hue_offset += CONFIG['color_cycle_speed']

    def get_edges(self, width, height):
        try:
            projected = []
            for v in self.vertices_4d:
                x, y, z, w = v
                c, s = math.cos(self.angle_xy), math.sin(self.angle_xy)
                x, w = x * c - w * s, x * s + w * c
                c, s = math.cos(self.angle_zw), math.sin(self.angle_zw)
                z, w = z * c - w * s, z * s + w * c
                c, s = math.cos(self.angle_xz), math.sin(self.angle_xz)
                x, z = x * c - z * s, x * s + z * c
                c, s = math.cos(self.angle_yz), math.sin(self.angle_yz)
                y, z = y * c - z * s, y * s + z * c
                d4, d3 = 3.0, 4.0
                s4 = d4 / (d4 - w)
                x3, y3, z3 = x * s4, y * s4, z * s4
                s3 = d3 / (d3 - z3)
                size = min(width, height) * 0.14
                projected.append((width/2 + x3*s3*size, height/2 - y3*s3*size, (w+1)/2, s4*s3))
            edges = []
            for idx, (i, j) in enumerate(self.edges):
                x1, y1, d1, s1 = projected[i]
                x2, y2, d2, s2 = projected[j]
                hue = (idx / len(self.edges) + self.hue_offset) % 1.0
                color = hsv_to_hex(hue, 0.7 + (d1+d2)/4*0.3, 0.6 + (d1+d2)/4*0.4)
                lw = max(1, int(CONFIG['line_width'] * (s1 + s2) / 2))
                edges.append((x1, y1, x2, y2, color, lw))
            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 2: Sacred Geometry Mandala
# =============================================================================
class SacredMandala:
    """Flower of Life inspired mandala with rotating petals"""
    name = "Sacred Mandala"
    num_lines = 250

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n_petals = random.choice([6, 8, 12])
        self.n_layers = 5
        self.angles = [random.random() * math.pi for _ in range(self.n_layers)]

    def update(self):
        self.t += 0.008
        for i in range(self.n_layers):
            self.angles[i] += 0.006 * (1 if i % 2 == 0 else -1) * (1 + i * 0.3)
        self.hue_offset += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            max_r = min(width, height) * 0.4
            edges = []
            pulse = 1 + 0.05 * math.sin(self.t * 0.5)

            for layer in range(self.n_layers):
                r = (layer + 1) / self.n_layers * max_r * pulse
                hue = (layer / self.n_layers + self.hue_offset) % 1.0
                pts = []
                for i in range(self.n_petals):
                    a = self.angles[layer] + i * 2 * math.pi / self.n_petals
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

                # Petal outline
                for i in range(self.n_petals):
                    edges.append((pts[i][0], pts[i][1], pts[(i+1) % self.n_petals][0], pts[(i+1) % self.n_petals][1],
                                 hsv_to_hex((hue + i * 0.05) % 1, 0.8, 0.7), 2))

                # Flower connections - every other petal
                for i in range(self.n_petals):
                    j = (i + 2) % self.n_petals
                    edges.append((pts[i][0], pts[i][1], pts[j][0], pts[j][1],
                                 hsv_to_hex((hue + 0.3) % 1, 0.6, 0.5), 1))

            # Inter-layer spokes
            for layer in range(self.n_layers - 1):
                r1 = (layer + 1) / self.n_layers * max_r * pulse
                r2 = (layer + 2) / self.n_layers * max_r * pulse
                hue = ((layer + 0.5) / self.n_layers + self.hue_offset) % 1.0
                for i in range(self.n_petals):
                    a1 = self.angles[layer] + i * 2 * math.pi / self.n_petals
                    a2 = self.angles[layer + 1] + i * 2 * math.pi / self.n_petals
                    x1, y1 = cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)
                    x2, y2 = cx + r2 * math.cos(a2), cy + r2 * math.sin(a2)
                    edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.7, 0.55), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 3: Spiraling Icosahedron
# =============================================================================
class SpiralingIcosahedron:
    """3D Icosahedron with spiraling vertex trails"""
    name = "Spiraling Icosahedron"
    num_lines = 350

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = self.angle_y = self.angle_z = 0.0
        phi = (1 + math.sqrt(5)) / 2
        self.base_verts = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
        ]
        self.edges_idx = [
            (0,1), (0,4), (0,5), (0,8), (0,9), (1,6), (1,7), (1,8), (1,9),
            (2,3), (2,4), (2,5), (2,10), (2,11), (3,6), (3,7), (3,10), (3,11),
            (4,5), (4,8), (4,10), (5,9), (5,11), (6,7), (6,8), (6,10), (7,9), (7,11), (8,10), (9,11)
        ]
        self.trails = [[] for _ in range(12)]

    def update(self):
        self.t += 0.02
        self.angle_x += 0.007
        self.angle_y += 0.009
        self.angle_z += 0.005
        self.hue_offset += 0.004

    def _rotate(self, v):
        x, y, z = v
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        x, z = x * cy - z * sy, x * sy + z * cy
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        y, z = y * cx - z * sx, y * sx + z * cx
        cz, sz = math.cos(self.angle_z), math.sin(self.angle_z)
        x, y = x * cz - y * sz, x * sz + y * cz
        return x, y, z

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.18
            edges = []

            # Rotate and project vertices
            projected = []
            for i, v in enumerate(self.base_verts):
                x, y, z = self._rotate(v)
                d = 5.0
                s = d / (d - z)
                px, py = cx + x * s * size, cy - y * s * size
                projected.append((px, py, z, s))
                # Add to trail
                self.trails[i].append((px, py))
                if len(self.trails[i]) > 30:
                    self.trails[i] = self.trails[i][-30:]

            # Draw edges
            for idx, (i, j) in enumerate(self.edges_idx):
                hue = (idx / len(self.edges_idx) + self.hue_offset) % 1.0
                x1, y1, z1, s1 = projected[i]
                x2, y2, z2, s2 = projected[j]
                lw = max(1, int(2.5 * (s1 + s2) / 2))
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.8, 0.7), lw))

            # Draw trails
            for vi, trail in enumerate(self.trails):
                for ti in range(1, len(trail)):
                    t = ti / len(trail)
                    hue = (vi / 12 + self.hue_offset + 0.5) % 1.0
                    edges.append((trail[ti-1][0], trail[ti-1][1], trail[ti][0], trail[ti][1],
                                 hsv_to_hex(hue, 0.6, t * 0.5), 1))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 4: Lotus Mandala
# =============================================================================
class LotusMandala:
    """Multi-layered lotus flower mandala"""
    name = "Lotus Mandala"
    num_lines = 300

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n_petals = random.choice([8, 12, 16])
        self.petal_layers = 4
        self.bloom = 0.5

    def update(self):
        self.t += 0.006
        self.bloom = 0.5 + 0.3 * math.sin(self.t * 0.3)
        self.hue_offset += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            max_r = min(width, height) * 0.38
            edges = []

            for layer in range(self.petal_layers):
                layer_r = (layer + 1) / self.petal_layers * max_r
                petal_w = layer_r * 0.4 * self.bloom
                rot = self.t * (0.3 if layer % 2 == 0 else -0.3) + layer * math.pi / self.n_petals
                hue = (layer / self.petal_layers + self.hue_offset) % 1.0

                for p in range(self.n_petals):
                    a = rot + p * 2 * math.pi / self.n_petals
                    # Petal tip
                    tip_x = cx + layer_r * math.cos(a)
                    tip_y = cy + layer_r * math.sin(a)
                    # Petal sides
                    side_r = layer_r * 0.5
                    left_a = a - petal_w / layer_r
                    right_a = a + petal_w / layer_r
                    left_x = cx + side_r * math.cos(left_a)
                    left_y = cy + side_r * math.sin(left_a)
                    right_x = cx + side_r * math.cos(right_a)
                    right_y = cy + side_r * math.sin(right_a)
                    # Draw petal
                    ph = (hue + p * 0.03) % 1.0
                    edges.append((left_x, left_y, tip_x, tip_y, hsv_to_hex(ph, 0.8, 0.7), 2))
                    edges.append((right_x, right_y, tip_x, tip_y, hsv_to_hex(ph, 0.8, 0.7), 2))
                    edges.append((left_x, left_y, right_x, right_y, hsv_to_hex((ph + 0.1) % 1, 0.6, 0.5), 1))

            # Center circle
            n_center = 12
            center_r = max_r * 0.15
            for i in range(n_center):
                a1 = i * 2 * math.pi / n_center
                a2 = (i + 1) * 2 * math.pi / n_center
                x1 = cx + center_r * math.cos(a1)
                y1 = cy + center_r * math.sin(a1)
                x2 = cx + center_r * math.cos(a2)
                y2 = cy + center_r * math.sin(a2)
                edges.append((x1, y1, x2, y2, hsv_to_hex((self.hue_offset + 0.5) % 1, 0.9, 0.8), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 5: Rotating Dodecahedron
# =============================================================================
class RotatingDodecahedron:
    """3D Dodecahedron with smooth rotation"""
    name = "Dodecahedron"
    num_lines = 30

    def __init__(self):
        self.angle_x = self.angle_y = self.angle_z = 0.0
        self.hue_offset = 0.0
        phi = (1 + math.sqrt(5)) / 2
        ip = 1 / phi
        self.verts = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
            (0, ip, phi), (0, ip, -phi), (0, -ip, phi), (0, -ip, -phi),
            (ip, phi, 0), (ip, -phi, 0), (-ip, phi, 0), (-ip, -phi, 0),
            (phi, 0, ip), (phi, 0, -ip), (-phi, 0, ip), (-phi, 0, -ip)
        ]
        self.edges_idx = [
            (0,8), (0,12), (0,16), (1,9), (1,12), (1,17), (2,10), (2,13), (2,16),
            (3,11), (3,13), (3,17), (4,8), (4,14), (4,18), (5,9), (5,14), (5,19),
            (6,10), (6,15), (6,18), (7,11), (7,15), (7,19), (8,10), (9,11),
            (12,14), (13,15), (16,17), (18,19)
        ]

    def update(self):
        self.angle_x += 0.006
        self.angle_y += 0.008
        self.angle_z += 0.004
        self.hue_offset += 0.004

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.15
            edges = []

            projected = []
            for v in self.verts:
                x, y, z = v
                cy_, sy = math.cos(self.angle_y), math.sin(self.angle_y)
                x, z = x * cy_ - z * sy, x * sy + z * cy_
                cx_, sx = math.cos(self.angle_x), math.sin(self.angle_x)
                y, z = y * cx_ - z * sx, y * sx + z * cx_
                cz, sz = math.cos(self.angle_z), math.sin(self.angle_z)
                x, y = x * cz - y * sz, x * sz + y * cz
                d = 5.0
                s = d / (d - z)
                projected.append((cx + x * s * size, cy - y * s * size, z, s))

            for idx, (i, j) in enumerate(self.edges_idx):
                x1, y1, z1, s1 = projected[i]
                x2, y2, z2, s2 = projected[j]
                hue = (idx / len(self.edges_idx) + self.hue_offset) % 1.0
                depth = (z1 + z2 + 4) / 8
                lw = max(1, int(3 * (s1 + s2) / 2))
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.75, 0.5 + depth * 0.4), lw))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 6: Sri Yantra Mandala
# =============================================================================
class SriYantra:
    """Sri Yantra sacred geometry with rotating triangles"""
    name = "Sri Yantra"
    num_lines = 200

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.outer_rot = 0.0
        self.inner_rot = 0.0

    def update(self):
        self.t += 0.008
        self.outer_rot += 0.003
        self.inner_rot -= 0.005
        self.hue_offset += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            max_r = min(width, height) * 0.38
            edges = []

            # Outer square (Bhupura)
            sq_r = max_r * 0.95
            for i in range(4):
                a1 = self.outer_rot + i * math.pi / 2 + math.pi / 4
                a2 = self.outer_rot + (i + 1) * math.pi / 2 + math.pi / 4
                x1, y1 = cx + sq_r * math.cos(a1), cy + sq_r * math.sin(a1)
                x2, y2 = cx + sq_r * math.cos(a2), cy + sq_r * math.sin(a2)
                edges.append((x1, y1, x2, y2, hsv_to_hex(self.hue_offset, 0.6, 0.5), 2))

            # Concentric circles
            for i in range(3):
                r = max_r * (0.85 - i * 0.1)
                n_seg = 36
                for j in range(n_seg):
                    a1 = j * 2 * math.pi / n_seg
                    a2 = (j + 1) * 2 * math.pi / n_seg
                    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
                    x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
                    hue = (i * 0.15 + self.hue_offset) % 1.0
                    edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.5, 0.45), 1))

            # Interlocking triangles (9 triangles - 4 up, 5 down)
            tri_scales = [0.7, 0.55, 0.4, 0.25]
            for ti, scale in enumerate(tri_scales):
                r = max_r * scale
                # Upward triangle
                up_rot = self.inner_rot + ti * 0.1
                hue = (ti * 0.12 + self.hue_offset + 0.3) % 1.0
                pts_up = []
                for i in range(3):
                    a = up_rot + i * 2 * math.pi / 3 - math.pi / 2
                    pts_up.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                for i in range(3):
                    edges.append((pts_up[i][0], pts_up[i][1], pts_up[(i+1)%3][0], pts_up[(i+1)%3][1],
                                 hsv_to_hex(hue, 0.8, 0.7), 2))

                # Downward triangle
                down_rot = -self.inner_rot - ti * 0.1
                hue2 = (ti * 0.12 + self.hue_offset + 0.6) % 1.0
                pts_down = []
                for i in range(3):
                    a = down_rot + i * 2 * math.pi / 3 + math.pi / 2
                    pts_down.append((cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a)))
                for i in range(3):
                    edges.append((pts_down[i][0], pts_down[i][1], pts_down[(i+1)%3][0], pts_down[(i+1)%3][1],
                                 hsv_to_hex(hue2, 0.8, 0.7), 2))

            # Central bindu (point)
            bindu_r = max_r * 0.05
            for i in range(8):
                a1 = i * math.pi / 4
                a2 = (i + 1) * math.pi / 4
                x1, y1 = cx + bindu_r * math.cos(a1), cy + bindu_r * math.sin(a1)
                x2, y2 = cx + bindu_r * math.cos(a2), cy + bindu_r * math.sin(a2)
                edges.append((x1, y1, x2, y2, hsv_to_hex((self.hue_offset + 0.5) % 1, 0.9, 0.9), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 7: Stellated Octahedron (Merkaba)
# =============================================================================
class Merkaba:
    """3D Merkaba / Star Tetrahedron"""
    name = "Merkaba"
    num_lines = 24

    def __init__(self):
        self.angle_x = self.angle_y = self.angle_z = 0.0
        self.hue_offset = 0.0
        s = 1.0
        # Two interlocking tetrahedra
        self.tetra1 = [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
        self.tetra2 = [(-s, -s, -s), (-s, s, s), (s, -s, s), (s, s, -s)]
        self.edges1 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        self.edges2 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    def update(self):
        self.angle_x += 0.008
        self.angle_y += 0.01
        self.angle_z += 0.006
        self.hue_offset += 0.004

    def _rotate(self, v):
        x, y, z = v
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        x, z = x * cy - z * sy, x * sy + z * cy
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        y, z = y * cx - z * sx, y * sx + z * cx
        cz, sz = math.cos(self.angle_z), math.sin(self.angle_z)
        x, y = x * cz - y * sz, x * sz + y * cz
        return x, y, z

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.2
            edges = []

            for tetra, edges_idx, hue_base in [(self.tetra1, self.edges1, 0), (self.tetra2, self.edges2, 0.5)]:
                projected = []
                for v in tetra:
                    x, y, z = self._rotate(v)
                    d = 4.0
                    s = d / (d - z)
                    projected.append((cx + x * s * size, cy - y * s * size, z, s))

                for idx, (i, j) in enumerate(edges_idx):
                    x1, y1, z1, s1 = projected[i]
                    x2, y2, z2, s2 = projected[j]
                    hue = (hue_base + idx * 0.08 + self.hue_offset) % 1.0
                    lw = max(2, int(3.5 * (s1 + s2) / 2))
                    edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.8, 0.75), lw))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 8: Spiral Galaxy Mandala
# =============================================================================
class GalaxyMandala:
    """Spiral arms forming a galaxy-like mandala"""
    name = "Galaxy Mandala"
    num_lines = 400

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n_arms = random.choice([2, 3, 4, 5])
        self.points_per_arm = 80

    def update(self):
        self.t += 0.008
        self.hue_offset += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            max_r = min(width, height) * 0.42
            edges = []

            for arm in range(self.n_arms):
                arm_offset = arm * 2 * math.pi / self.n_arms
                pts = []
                for i in range(self.points_per_arm):
                    t = i / self.points_per_arm
                    r = t * max_r
                    # Logarithmic spiral with rotation
                    a = arm_offset + t * 4 * math.pi + self.t
                    x = cx + r * math.cos(a)
                    y = cy + r * math.sin(a)
                    pts.append((x, y))

                hue_base = (arm / self.n_arms + self.hue_offset) % 1.0
                for i in range(1, len(pts)):
                    t = i / len(pts)
                    hue = (hue_base + t * 0.3) % 1.0
                    edges.append((pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1],
                                 hsv_to_hex(hue, 0.7, 0.4 + t * 0.5), 2))

            # Central bulge
            n_center = 20
            for i in range(n_center):
                a1 = self.t * 2 + i * 2 * math.pi / n_center
                a2 = self.t * 2 + (i + 1) * 2 * math.pi / n_center
                r = max_r * 0.08
                x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
                x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
                edges.append((x1, y1, x2, y2, hsv_to_hex((self.hue_offset + 0.5) % 1, 0.8, 0.9), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 9: Kaleidoscope Mandala
# =============================================================================
class KaleidoscopeMandala:
    """Dynamic kaleidoscope with mirrored segments"""
    name = "Kaleidoscope"
    num_lines = 300

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n_mirrors = random.choice([6, 8, 10, 12])
        self.shapes = []
        for _ in range(15):
            self.shapes.append({
                'r': random.random() * 0.7 + 0.1,
                'a': random.random() * math.pi * 2,
                'dr': (random.random() - 0.5) * 0.003,
                'da': (random.random() - 0.5) * 0.02,
                'size': random.random() * 0.08 + 0.03,
                'sides': random.choice([3, 4, 5, 6])
            })

    def update(self):
        self.t += 0.008
        self.hue_offset += 0.004
        for s in self.shapes:
            s['r'] += s['dr']
            s['a'] += s['da']
            if s['r'] < 0.1 or s['r'] > 0.8:
                s['dr'] *= -1

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            max_r = min(width, height) * 0.42
            edges = []

            for si, shape in enumerate(self.shapes):
                hue_base = (si / len(self.shapes) + self.hue_offset) % 1.0
                # Draw shape in each mirror segment
                for m in range(self.n_mirrors):
                    mirror_a = m * 2 * math.pi / self.n_mirrors
                    # Shape position
                    sr = shape['r'] * max_r
                    sa = shape['a'] + mirror_a
                    scx = cx + sr * math.cos(sa)
                    scy = cy + sr * math.sin(sa)
                    # Draw polygon
                    sz = shape['size'] * max_r
                    pts = []
                    for i in range(shape['sides']):
                        pa = self.t + i * 2 * math.pi / shape['sides'] + mirror_a
                        pts.append((scx + sz * math.cos(pa), scy + sz * math.sin(pa)))
                    for i in range(shape['sides']):
                        hue = (hue_base + m * 0.02) % 1.0
                        edges.append((pts[i][0], pts[i][1], pts[(i+1) % shape['sides']][0], pts[(i+1) % shape['sides']][1],
                                     hsv_to_hex(hue, 0.8, 0.7), 2))

            # Mirror lines from center
            for m in range(self.n_mirrors):
                a = m * 2 * math.pi / self.n_mirrors + self.t * 0.1
                x2 = cx + max_r * 0.95 * math.cos(a)
                y2 = cy + max_r * 0.95 * math.sin(a)
                hue = (m / self.n_mirrors + self.hue_offset) % 1.0
                edges.append((cx, cy, x2, y2, hsv_to_hex(hue, 0.4, 0.3), 1))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 10: Stellated Dodecahedron (Great Stellated)
# =============================================================================
class StellatedDodecahedron:
    """3D Great Stellated Dodecahedron"""
    name = "Stellated Dodecahedron"
    num_lines = 90

    def __init__(self):
        self.angle_x = self.angle_y = self.angle_z = 0.0
        self.hue_offset = 0.0
        phi = (1 + math.sqrt(5)) / 2
        # Icosahedron vertices (dual of dodecahedron)
        self.verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                self.verts.append((0, s1, s2 * phi))
                self.verts.append((s1, s2 * phi, 0))
                self.verts.append((s2 * phi, 0, s1))
        # Stellate by extending vertices
        self.stell_factor = phi * phi
        self.spike_verts = [(v[0] * self.stell_factor, v[1] * self.stell_factor, v[2] * self.stell_factor) for v in self.verts]
        # Edges: connect spikes to neighbors
        self.edges_idx = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = sum((self.verts[i][k] - self.verts[j][k])**2 for k in range(3))
                if d < 5:  # Adjacent vertices
                    self.edges_idx.append((i, j))

    def update(self):
        self.angle_x += 0.006
        self.angle_y += 0.008
        self.angle_z += 0.004
        self.hue_offset += 0.004

    def _rotate(self, v):
        x, y, z = v
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        x, z = x * cy - z * sy, x * sy + z * cy
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        y, z = y * cx - z * sx, y * sx + z * cx
        cz, sz = math.cos(self.angle_z), math.sin(self.angle_z)
        x, y = x * cz - y * sz, x * sz + y * cz
        return x, y, z

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.08
            edges = []

            # Project spike vertices
            proj_spikes = []
            for v in self.spike_verts:
                x, y, z = self._rotate(v)
                d = 6.0
                s = d / (d - z)
                proj_spikes.append((cx + x * s * size, cy - y * s * size, z, s))

            # Project base vertices
            proj_base = []
            for v in self.verts:
                x, y, z = self._rotate(v)
                d = 6.0
                s = d / (d - z)
                proj_base.append((cx + x * s * size, cy - y * s * size, z, s))

            # Draw spike edges
            for idx, (i, j) in enumerate(self.edges_idx):
                x1, y1, z1, s1 = proj_spikes[i]
                x2, y2, z2, s2 = proj_spikes[j]
                hue = (idx / len(self.edges_idx) + self.hue_offset) % 1.0
                lw = max(2, int(3 * (s1 + s2) / 2))
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.8, 0.7), lw))

            # Draw connections from base to spikes
            for i in range(12):
                x1, y1, _, s1 = proj_base[i]
                x2, y2, _, s2 = proj_spikes[i]
                hue = (i / 12 + self.hue_offset + 0.5) % 1.0
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.7, 0.55), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 11: Harmonic Rose Mandala
# =============================================================================
class HarmonicRose:
    """Rose curves (rhodonea) forming harmonic mandala"""
    name = "Harmonic Rose"
    num_lines = 400

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n = random.choice([3, 4, 5, 6, 7])  # Petal parameter
        self.d = random.choice([2, 3, 5, 7])  # Density parameter
        self.points = []
        self.rot = 0.0

    def update(self):
        self.t += 0.03
        self.rot += 0.003
        self.hue_offset += 0.003

        # Rose curve: r = cos(n/d * theta)
        k = self.n / self.d
        r = math.cos(k * self.t)
        x = r * math.cos(self.t)
        y = r * math.sin(self.t)
        self.points.append((x, y))
        if len(self.points) > 380:
            self.points = self.points[-380:]

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.38
            edges = []
            cos_r, sin_r = math.cos(self.rot), math.sin(self.rot)

            for i in range(1, len(self.points)):
                x1, y1 = self.points[i - 1]
                x2, y2 = self.points[i]

                # Rotate
                x1, y1 = x1 * cos_r - y1 * sin_r, x1 * sin_r + y1 * cos_r
                x2, y2 = x2 * cos_r - y2 * sin_r, x2 * sin_r + y2 * cos_r

                px1, py1 = cx + x1 * size, cy - y1 * size
                px2, py2 = cx + x2 * size, cy - y2 * size

                t = i / len(self.points)
                hue = (t * 0.8 + self.hue_offset) % 1.0
                edges.append((px1, py1, px2, py2, hsv_to_hex(hue, 0.8, 0.4 + t * 0.55), 2))

            return edges
        except Exception:
            return []


# =============================================================================
# STAR FIELD (Background)
# =============================================================================
class StarField:
    """Background stars with color cycling"""

    def __init__(self, width, height):
        self.width, self.height = max(1, width), max(1, height)
        self.hue_offset = 0.0
        self.stars = [{'x': random.randint(0, self.width), 'y': random.randint(0, self.height),
                       'z': random.random() * 2 + 0.5, 'brightness': random.uniform(0.3, 0.8),
                       'hue': random.random()} for _ in range(CONFIG['num_stars'])]

    def update(self):
        self.hue_offset += CONFIG['color_cycle_speed'] * 0.3
        for star in self.stars:
            star['x'] -= CONFIG['star_speed'] * star['z']
            if star['x'] < 0:
                star['x'] = self.width
                star['y'] = random.randint(0, max(1, self.height))
                star['hue'] = random.random()

    def resize(self, w, h):
        self.width, self.height = max(1, w), max(1, h)

    def get_stars(self):
        return [(s['x'], s['y'], hsv_to_hex((s['hue'] + self.hue_offset) % 1.0, 0.4, s['brightness']))
                for s in self.stars]


# =============================================================================
# MAIN SCREENSAVER
# =============================================================================
class Screensaver:
    """Multi-visualization screensaver - 11 Mandalas & 3D shapes"""

    VISUALIZATIONS = [
        Tesseract, SacredMandala, SpiralingIcosahedron, LotusMandala,
        RotatingDodecahedron, SriYantra, Merkaba, GalaxyMandala,
        KaleidoscopeMandala, StellatedDodecahedron, HarmonicRose
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mandala & 3D Screensaver")
        self.root.configure(bg=CONFIG['bg_color'])

        self.width, self.height = 900, 700
        self.root.geometry(f'{self.width}x{self.height}')
        self.root.minsize(400, 300)

        # Clock bar
        self.clock_frame = tk.Frame(self.root, bg='#000000', height=CONFIG['clock_height'])
        self.clock_frame.pack(fill=tk.X, side=tk.TOP)
        self.clock_frame.pack_propagate(False)

        self.clock_label = tk.Label(self.clock_frame, text="", font=('Arial', CONFIG['clock_font_size'], 'bold'),
                                    fg='#ffffff', bg='#000000')
        self.clock_label.pack(expand=True)

        self.name_label = tk.Label(self.clock_frame, text="", font=('Arial', 14),
                                   fg='#666666', bg='#000000')
        self.name_label.place(relx=0.98, rely=0.5, anchor='e')

        # Canvas
        self.canvas = tk.Canvas(self.root, bg=CONFIG['bg_color'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize
        self.stars = StarField(self.width, self.height)
        self.current_viz = None
        self.line_items = []
        self.star_items = [self.canvas.create_oval(0, 0, 1, 1, fill='', outline='')
                          for _ in range(CONFIG['num_stars'])]

        self._switch_visualization()

        # Bindings
        self.root.bind('<Escape>', lambda e: self.quit())
        self.root.bind('<space>', lambda e: self._switch_visualization())
        self.root.bind('<Configure>', self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.frame_time = 1000 // CONFIG['fps']
        self.running = True

        self.root.after(CONFIG['switch_interval'], self._auto_switch)

    def _switch_visualization(self):
        try:
            choices = [v for v in self.VISUALIZATIONS if not isinstance(self.current_viz, v)]
            VizClass = random.choice(choices) if choices else random.choice(self.VISUALIZATIONS)
            self.current_viz = VizClass()

            for item in self.line_items:
                self.canvas.delete(item)
            self.line_items = [self.canvas.create_line(0, 0, 0, 0, fill='', width=1)
                              for _ in range(self.current_viz.num_lines)]

            self.name_label.config(text=self.current_viz.name)
            self.root.title(f"Screensaver - {self.current_viz.name}")
        except Exception:
            pass

    def _auto_switch(self):
        if self.running:
            self._switch_visualization()
            self.root.after(CONFIG['switch_interval'], self._auto_switch)

    def _on_resize(self, event):
        if event.widget == self.canvas:
            self.width, self.height = event.width, event.height
            self.stars.resize(self.width, self.height)

    def quit(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def draw(self):
        try:
            self.clock_label.config(text=datetime.now().strftime('%H:%M:%S'))

            s = CONFIG['star_size']
            for i, (x, y, color) in enumerate(self.stars.get_stars()):
                if i < len(self.star_items):
                    self.canvas.coords(self.star_items[i], x-s, y-s, x+s, y+s)
                    self.canvas.itemconfig(self.star_items[i], fill=color)

            edges = self.current_viz.get_edges(self.width, self.height) if self.current_viz else []
            for i, item in enumerate(self.line_items):
                if i < len(edges):
                    x1, y1, x2, y2, color, lw = edges[i]
                    self.canvas.coords(item, x1, y1, x2, y2)
                    self.canvas.itemconfig(item, fill=color, width=lw)
                else:
                    self.canvas.coords(item, 0, 0, 0, 0)
        except Exception:
            pass

    def run(self):
        def frame():
            if not self.running:
                return
            try:
                if self.current_viz:
                    self.current_viz.update()
                self.stars.update()
                self.draw()
            except Exception:
                pass
            self.root.after(self.frame_time, frame)

        frame()
        try:
            self.root.mainloop()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        Screensaver().run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
