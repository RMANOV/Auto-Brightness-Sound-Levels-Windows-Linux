#!/usr/bin/env python3
"""Spectral Geometry Screensaver 09 - Pure Tkinter"""

import math
import random
import time
import tkinter as tk
from datetime import datetime

CONFIG = {
    "fps": 30,
    "bg_color": "#06080c",
    "num_stars": 40,
    "star_speed": 0.3,
    "star_size": 3,
    "color_cycle_speed": 0.004,
    "clock_height": 80,
    "clock_font_size": 48,
    "line_width": 3,
    "hilbert_order": 4,  # 256 points
    "hilbert_line_width": 2,
    "min_follow_distance": 120,  # creatures never closer than this to cursor
    "follow_ease": 0.03,  # lerp factor for cursor following
    "wander_ease": 0.01,  # lerp factor for random wandering
    "cursor_timeout": 3.0,  # seconds before cursor considered absent
    "rotation_speed": 0.008,
    "max_lines": 3200,  # pre-allocated line pool
    "switch_interval": 7 * 60 * 1000,  # 7 minutes in ms
}


# =============================================================================
# UTILITY
# =============================================================================


def hsv_to_hex(h, s, v):
    try:
        h = h % 1.0
        i = int(h * 6)
        f = (h * 6) - i
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
        return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
    except Exception:
        return "#8080ff"


def rotate_3d(x, y, z, ax=0.0, ay=0.0, az=0.0):
    if ay:
        cy, sy = math.cos(ay), math.sin(ay)
        x, z = x * cy - z * sy, x * sy + z * cy
    if ax:
        cx, sx = math.cos(ax), math.sin(ax)
        y, z = y * cx - z * sx, y * sx + z * cx
    if az:
        cz, sz = math.cos(az), math.sin(az)
        x, y = x * cz - y * sz, x * sz + y * cz
    return x, y, z


def project_3d(x, y, z, cx, cy, size, dist):
    denom = max(0.2, dist - z)
    scale = dist / denom
    return cx + x * scale * size, cy - y * scale * size, z, scale


# =============================================================================
# STAR FIELD
# =============================================================================


class StarField:
    def __init__(self, width, height):
        self.width, self.height = max(1, width), max(1, height)
        self.hue_offset = 0.0
        self.stars = [
            {
                "x": random.randint(0, self.width),
                "y": random.randint(0, self.height),
                "z": random.random() * 2 + 0.5,
                "brightness": random.uniform(0.3, 0.8),
                "hue": random.random(),
            }
            for _ in range(CONFIG["num_stars"])
        ]

    def update(self):
        self.hue_offset += CONFIG["color_cycle_speed"] * 0.3
        for star in self.stars:
            star["x"] -= CONFIG["star_speed"] * star["z"]
            if star["x"] < 0:
                star["x"] = self.width
                star["y"] = random.randint(0, max(1, self.height))
                star["hue"] = random.random()

    def resize(self, w, h):
        self.width, self.height = max(1, w), max(1, h)

    def get_stars(self):
        return [
            (
                s["x"],
                s["y"],
                hsv_to_hex((s["hue"] + self.hue_offset) % 1.0, 0.4, s["brightness"]),
            )
            for s in self.stars
        ]


# =============================================================================
# 18 VISUALIZATIONS FROM VERSION 05
# =============================================================================


class Tesseract:
    """4D Hypercube projection"""

    name = "4D Tesseract"
    num_lines = 32

    def __init__(self):
        self.angle_xy = self.angle_zw = self.angle_xz = self.angle_yz = 0.0
        self.hue_offset = 0.0
        self.vertices_4d = [
            [x, y, z, w]
            for x in [-1, 1]
            for y in [-1, 1]
            for z in [-1, 1]
            for w in [-1, 1]
        ]
        self.edges = [
            (i, j)
            for i in range(16)
            for j in range(i + 1, 16)
            if sum(
                1 for k in range(4) if self.vertices_4d[i][k] != self.vertices_4d[j][k]
            )
            == 1
        ]

    def update(self):
        self.angle_xy += CONFIG["rotation_speed"] * 1.5
        self.angle_zw += CONFIG["rotation_speed"] * 1.05
        self.angle_xz += CONFIG["rotation_speed"]
        self.angle_yz += CONFIG["rotation_speed"] * 0.5
        self.hue_offset += CONFIG["color_cycle_speed"]

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
                projected.append(
                    (
                        width / 2 + x3 * s3 * size,
                        height / 2 - y3 * s3 * size,
                        (w + 1) / 2,
                        s4 * s3,
                    )
                )
            edges = []
            for idx, (i, j) in enumerate(self.edges):
                x1, y1, d1, s1 = projected[i]
                x2, y2, d2, s2 = projected[j]
                hue = (idx / len(self.edges) + self.hue_offset) % 1.0
                color = hsv_to_hex(
                    hue, 0.7 + (d1 + d2) / 4 * 0.3, 0.6 + (d1 + d2) / 4 * 0.4
                )
                lw = max(1, int(CONFIG["line_width"] * (s1 + s2) / 2))
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
                    edges.append(
                        (
                            pts[i][0],
                            pts[i][1],
                            pts[(i + 1) % self.n_petals][0],
                            pts[(i + 1) % self.n_petals][1],
                            hsv_to_hex((hue + i * 0.05) % 1, 0.8, 0.7),
                            2,
                        )
                    )

                # Flower connections - every other petal
                for i in range(self.n_petals):
                    j = (i + 2) % self.n_petals
                    edges.append(
                        (
                            pts[i][0],
                            pts[i][1],
                            pts[j][0],
                            pts[j][1],
                            hsv_to_hex((hue + 0.3) % 1, 0.6, 0.5),
                            1,
                        )
                    )

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
            (0, 1, phi),
            (0, -1, phi),
            (0, 1, -phi),
            (0, -1, -phi),
            (1, phi, 0),
            (-1, phi, 0),
            (1, -phi, 0),
            (-1, -phi, 0),
            (phi, 0, 1),
            (-phi, 0, 1),
            (phi, 0, -1),
            (-phi, 0, -1),
        ]
        self.edges_idx = [
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 10),
            (2, 11),
            (3, 6),
            (3, 7),
            (3, 10),
            (3, 11),
            (4, 5),
            (4, 8),
            (4, 10),
            (5, 9),
            (5, 11),
            (6, 7),
            (6, 8),
            (6, 10),
            (7, 9),
            (7, 11),
            (8, 10),
            (9, 11),
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
                    edges.append(
                        (
                            trail[ti - 1][0],
                            trail[ti - 1][1],
                            trail[ti][0],
                            trail[ti][1],
                            hsv_to_hex(hue, 0.6, t * 0.5),
                            1,
                        )
                    )

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
                rot = (
                    self.t * (0.3 if layer % 2 == 0 else -0.3)
                    + layer * math.pi / self.n_petals
                )
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
                    edges.append(
                        (left_x, left_y, tip_x, tip_y, hsv_to_hex(ph, 0.8, 0.7), 2)
                    )
                    edges.append(
                        (right_x, right_y, tip_x, tip_y, hsv_to_hex(ph, 0.8, 0.7), 2)
                    )
                    edges.append(
                        (
                            left_x,
                            left_y,
                            right_x,
                            right_y,
                            hsv_to_hex((ph + 0.1) % 1, 0.6, 0.5),
                            1,
                        )
                    )

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
                edges.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        hsv_to_hex((self.hue_offset + 0.5) % 1, 0.9, 0.8),
                        2,
                    )
                )

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
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
            (0, ip, phi),
            (0, ip, -phi),
            (0, -ip, phi),
            (0, -ip, -phi),
            (ip, phi, 0),
            (ip, -phi, 0),
            (-ip, phi, 0),
            (-ip, -phi, 0),
            (phi, 0, ip),
            (phi, 0, -ip),
            (-phi, 0, ip),
            (-phi, 0, -ip),
        ]
        self.edges_idx = [
            (0, 8),
            (0, 12),
            (0, 16),
            (1, 9),
            (1, 12),
            (1, 17),
            (2, 10),
            (2, 13),
            (2, 16),
            (3, 11),
            (3, 13),
            (3, 17),
            (4, 8),
            (4, 14),
            (4, 18),
            (5, 9),
            (5, 14),
            (5, 19),
            (6, 10),
            (6, 15),
            (6, 18),
            (7, 11),
            (7, 15),
            (7, 19),
            (8, 10),
            (9, 11),
            (12, 14),
            (13, 15),
            (16, 17),
            (18, 19),
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
                edges.append(
                    (x1, y1, x2, y2, hsv_to_hex(hue, 0.75, 0.5 + depth * 0.4), lw)
                )

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
                    edges.append(
                        (
                            pts_up[i][0],
                            pts_up[i][1],
                            pts_up[(i + 1) % 3][0],
                            pts_up[(i + 1) % 3][1],
                            hsv_to_hex(hue, 0.8, 0.7),
                            2,
                        )
                    )

                # Downward triangle
                down_rot = -self.inner_rot - ti * 0.1
                hue2 = (ti * 0.12 + self.hue_offset + 0.6) % 1.0
                pts_down = []
                for i in range(3):
                    a = down_rot + i * 2 * math.pi / 3 + math.pi / 2
                    pts_down.append(
                        (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))
                    )
                for i in range(3):
                    edges.append(
                        (
                            pts_down[i][0],
                            pts_down[i][1],
                            pts_down[(i + 1) % 3][0],
                            pts_down[(i + 1) % 3][1],
                            hsv_to_hex(hue2, 0.8, 0.7),
                            2,
                        )
                    )

            # Central bindu (point)
            bindu_r = max_r * 0.05
            for i in range(8):
                a1 = i * math.pi / 4
                a2 = (i + 1) * math.pi / 4
                x1, y1 = cx + bindu_r * math.cos(a1), cy + bindu_r * math.sin(a1)
                x2, y2 = cx + bindu_r * math.cos(a2), cy + bindu_r * math.sin(a2)
                edges.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        hsv_to_hex((self.hue_offset + 0.5) % 1, 0.9, 0.9),
                        2,
                    )
                )

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
        self.edges1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.edges2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

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

            for tetra, edges_idx, hue_base in [
                (self.tetra1, self.edges1, 0),
                (self.tetra2, self.edges2, 0.5),
            ]:
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
# VISUALIZATION 8: 4D 24-Cell (Icositetrachoron)
# =============================================================================
class Cell24:
    """4D polytope with 24 octahedral cells - one of the most beautiful 4D shapes"""

    name = "4D 24-Cell"
    num_lines = 96

    def __init__(self):
        self.angle_xy = self.angle_xz = self.angle_xw = 0.0
        self.angle_yz = self.angle_yw = self.angle_zw = 0.0
        self.hue_offset = 0.0
        # 24 vertices: 8 from permutations of (±1,0,0,0) + 16 from (±½,±½,±½,±½)
        self.verts = []
        # Type A: permutations of (±1, 0, 0, 0)
        for i in range(4):
            for s in [-1, 1]:
                v = [0, 0, 0, 0]
                v[i] = s
                self.verts.append(tuple(v))
        # Type B: (±½, ±½, ±½, ±½)
        for s1 in [-0.5, 0.5]:
            for s2 in [-0.5, 0.5]:
                for s3 in [-0.5, 0.5]:
                    for s4 in [-0.5, 0.5]:
                        self.verts.append((s1, s2, s3, s4))
        # Edges: connect vertices at distance 1
        self.edges_idx = []
        for i in range(24):
            for j in range(i + 1, 24):
                d = sum((self.verts[i][k] - self.verts[j][k]) ** 2 for k in range(4))
                if abs(d - 1.0) < 0.01:
                    self.edges_idx.append((i, j))

    def update(self):
        self.angle_xy += 0.007
        self.angle_xz += 0.005
        self.angle_xw += 0.009
        self.angle_yz += 0.004
        self.angle_yw += 0.006
        self.angle_zw += 0.008
        self.hue_offset += 0.004

    def _rotate4d(self, v):
        x, y, z, w = v
        # XY rotation
        c, s = math.cos(self.angle_xy), math.sin(self.angle_xy)
        x, y = x * c - y * s, x * s + y * c
        # XZ rotation
        c, s = math.cos(self.angle_xz), math.sin(self.angle_xz)
        x, z = x * c - z * s, x * s + z * c
        # XW rotation
        c, s = math.cos(self.angle_xw), math.sin(self.angle_xw)
        x, w = x * c - w * s, x * s + w * c
        # YZ rotation
        c, s = math.cos(self.angle_yz), math.sin(self.angle_yz)
        y, z = y * c - z * s, y * s + z * c
        # YW rotation
        c, s = math.cos(self.angle_yw), math.sin(self.angle_yw)
        y, w = y * c - w * s, y * s + w * c
        # ZW rotation
        c, s = math.cos(self.angle_zw), math.sin(self.angle_zw)
        z, w = z * c - w * s, z * s + w * c
        return x, y, z, w

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.18
            edges = []

            projected = []
            for v in self.verts:
                x, y, z, w = self._rotate4d(v)
                # 4D to 3D perspective
                d4 = 2.5
                s4 = d4 / (d4 - w)
                x3, y3, z3 = x * s4, y * s4, z * s4
                # 3D to 2D perspective
                d3 = 4.0
                s3 = d3 / (d3 - z3)
                px = cx + x3 * s3 * size
                py = cy - y3 * s3 * size
                projected.append((px, py, w, s4 * s3))

            for idx, (i, j) in enumerate(self.edges_idx):
                x1, y1, w1, s1 = projected[i]
                x2, y2, w2, s2 = projected[j]
                hue = (idx / len(self.edges_idx) + self.hue_offset) % 1.0
                depth = (w1 + w2 + 2) / 4
                lw = max(1, int(3 * (s1 + s2) / 2))
                edges.append(
                    (x1, y1, x2, y2, hsv_to_hex(hue, 0.75, 0.4 + depth * 0.5), lw)
                )

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
            self.shapes.append(
                {
                    "r": random.random() * 0.7 + 0.1,
                    "a": random.random() * math.pi * 2,
                    "dr": (random.random() - 0.5) * 0.003,
                    "da": (random.random() - 0.5) * 0.02,
                    "size": random.random() * 0.08 + 0.03,
                    "sides": random.choice([3, 4, 5, 6]),
                }
            )

    def update(self):
        self.t += 0.008
        self.hue_offset += 0.004
        for s in self.shapes:
            s["r"] += s["dr"]
            s["a"] += s["da"]
            if s["r"] < 0.1 or s["r"] > 0.8:
                s["dr"] *= -1

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
                    sr = shape["r"] * max_r
                    sa = shape["a"] + mirror_a
                    scx = cx + sr * math.cos(sa)
                    scy = cy + sr * math.sin(sa)
                    # Draw polygon
                    sz = shape["size"] * max_r
                    pts = []
                    for i in range(shape["sides"]):
                        pa = self.t + i * 2 * math.pi / shape["sides"] + mirror_a
                        pts.append((scx + sz * math.cos(pa), scy + sz * math.sin(pa)))
                    for i in range(shape["sides"]):
                        hue = (hue_base + m * 0.02) % 1.0
                        edges.append(
                            (
                                pts[i][0],
                                pts[i][1],
                                pts[(i + 1) % shape["sides"]][0],
                                pts[(i + 1) % shape["sides"]][1],
                                hsv_to_hex(hue, 0.8, 0.7),
                                2,
                            )
                        )

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
        self.spike_verts = [
            (
                v[0] * self.stell_factor,
                v[1] * self.stell_factor,
                v[2] * self.stell_factor,
            )
            for v in self.verts
        ]
        # Edges: connect spikes to neighbors
        self.edges_idx = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = sum((self.verts[i][k] - self.verts[j][k]) ** 2 for k in range(3))
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
                edges.append(
                    (px1, py1, px2, py2, hsv_to_hex(hue, 0.8, 0.4 + t * 0.55), 2)
                )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 12: 4D 120-Cell (Hyperdodecahedron) - Partial
# =============================================================================
class Hyperdodecahedron:
    """Partial 4D 120-Cell - the most complex regular 4D polytope"""

    name = "4D Hyperdodecahedron"
    num_lines = 150

    def __init__(self):
        self.angle_xw = self.angle_yw = self.angle_zw = 0.0
        self.angle_xy = self.angle_xz = self.angle_yz = 0.0
        self.hue_offset = 0.0
        # 24-cell vertices: clean 4D polytope with 96 edges
        self.verts = []
        # 8 vertices: permutations of (±1, 0, 0, 0) — 4D cross-polytope
        for i in range(4):
            for s in [-1, 1]:
                v = [0, 0, 0, 0]
                v[i] = s
                self.verts.append(tuple(v))
        # 16 vertices: (±0.5, ±0.5, ±0.5, ±0.5) — half-tesseract
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    for s3 in [-1, 1]:
                        self.verts.append((s0 * 0.5, s1 * 0.5, s2 * 0.5, s3 * 0.5))
        # Build edges: 24-cell has uniform edge length = 1 (d² = 1.0)
        self.edges_idx = []
        for i in range(len(self.verts)):
            for j in range(i + 1, len(self.verts)):
                d = sum((self.verts[i][k] - self.verts[j][k]) ** 2 for k in range(4))
                if 0.9 < d < 1.1:  # Edge length exactly 1
                    self.edges_idx.append((i, j))

    def update(self):
        self.angle_xw += 0.006
        self.angle_yw += 0.008
        self.angle_zw += 0.005
        self.angle_xy += 0.004
        self.angle_xz += 0.003
        self.angle_yz += 0.007
        self.hue_offset += 0.004

    def _rotate4d(self, v):
        x, y, z, w = v
        for angle, (a, b) in [
            (self.angle_xy, (0, 1)),
            (self.angle_xz, (0, 2)),
            (self.angle_xw, (0, 3)),
            (self.angle_yz, (1, 2)),
            (self.angle_yw, (1, 3)),
            (self.angle_zw, (2, 3)),
        ]:
            c, s = math.cos(angle), math.sin(angle)
            coords = [x, y, z, w]
            coords[a], coords[b] = (
                coords[a] * c - coords[b] * s,
                coords[a] * s + coords[b] * c,
            )
            x, y, z, w = coords
        return x, y, z, w

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.14
            edges = []
            projected = []
            for v in self.verts:
                x, y, z, w = self._rotate4d(v)
                d4 = 4.0
                s4 = d4 / (d4 - w)
                x3, y3, z3 = x * s4, y * s4, z * s4
                d3 = 5.0
                s3 = d3 / (d3 - z3)
                projected.append((cx + x3 * s3 * size, cy - y3 * s3 * size, w, s4 * s3))

            for idx, (i, j) in enumerate(self.edges_idx):
                x1, y1, w1, s1 = projected[i]
                x2, y2, w2, s2 = projected[j]
                hue = (idx / max(1, len(self.edges_idx)) + self.hue_offset) % 1.0
                depth = (w1 + w2 + 6) / 12
                lw = max(1, int(2.5 * (s1 + s2) / 2))
                edges.append(
                    (x1, y1, x2, y2, hsv_to_hex(hue, 0.8, 0.5 + depth * 0.5), lw)
                )
            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 13: Cosmic Web (3D Neural Network)
# =============================================================================
class CosmicWeb:
    """3D network of interconnected nodes forming a cosmic web structure"""

    name = "Cosmic Web"
    num_lines = 400

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = self.angle_y = self.angle_z = 0.0
        # Generate random 3D nodes
        self.n_nodes = 35
        self.nodes = []
        for _ in range(self.n_nodes):
            self.nodes.append(
                {
                    "x": (random.random() - 0.5) * 2,
                    "y": (random.random() - 0.5) * 2,
                    "z": (random.random() - 0.5) * 2,
                    "vx": (random.random() - 0.5) * 0.002,
                    "vy": (random.random() - 0.5) * 0.002,
                    "vz": (random.random() - 0.5) * 0.002,
                    "pulse": random.random() * math.pi * 2,
                }
            )
        # Build edges - connect nearby nodes
        self.edges_idx = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                d = sum(
                    (self.nodes[i][k] - self.nodes[j][k]) ** 2 for k in ["x", "y", "z"]
                )
                if d < 0.8:
                    self.edges_idx.append((i, j))

    def update(self):
        self.t += 0.01
        self.angle_x += 0.004
        self.angle_y += 0.006
        self.angle_z += 0.003
        self.hue_offset += 0.003
        # Animate nodes
        for n in self.nodes:
            n["x"] += n["vx"]
            n["y"] += n["vy"]
            n["z"] += n["vz"]
            n["pulse"] += 0.05
            # Bounce off boundaries
            for k in ["x", "y", "z"]:
                if abs(n[k]) > 1:
                    n["v" + k] *= -1

    def _rotate(self, x, y, z):
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
            size = min(width, height) * 0.35
            edges = []

            # Project nodes
            projected = []
            for n in self.nodes:
                x, y, z = self._rotate(n["x"], n["y"], n["z"])
                d = 4.0
                s = d / (d - z)
                px, py = cx + x * s * size, cy - y * s * size
                pulse = 0.5 + 0.5 * math.sin(n["pulse"])
                projected.append((px, py, z, s, pulse))

            # Draw edges
            for idx, (i, j) in enumerate(self.edges_idx):
                x1, y1, z1, s1, p1 = projected[i]
                x2, y2, z2, s2, p2 = projected[j]
                hue = (idx / max(1, len(self.edges_idx)) + self.hue_offset) % 1.0
                brightness = 0.3 + 0.4 * (p1 + p2) / 2
                lw = max(1, int(2.5 * (s1 + s2) / 2))
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.75, brightness), lw))

            # Draw node halos
            for i, (px, py, z, s, pulse) in enumerate(projected):
                hue = (i / self.n_nodes + self.hue_offset + 0.5) % 1.0
                r = 8 * s * (0.7 + 0.3 * pulse)
                n_seg = 8
                for j in range(n_seg):
                    a1 = j * 2 * math.pi / n_seg + self.t
                    a2 = (j + 1) * 2 * math.pi / n_seg + self.t
                    x1, y1 = px + r * math.cos(a1), py + r * math.sin(a1)
                    x2, y2 = px + r * math.cos(a2), py + r * math.sin(a2)
                    edges.append(
                        (x1, y1, x2, y2, hsv_to_hex(hue, 0.9, 0.5 + 0.4 * pulse), 2)
                    )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 14: Uzumaki Spiral (3D Curlicue Fractal)
# =============================================================================
class UzumakiSpiral3D:
    """3D Curlicue Fractal spiral with hypnotic rotation"""

    name = "Uzumaki Spiral"
    num_lines = 450

    def __init__(self):
        self.angle_xz = 0.0  # Rotation around Y axis
        self.angle_yz = 0.0  # Tilt
        self.t = 0.0  # Time parameter for spiral evolution
        self.hue_offset = 0.0
        self.n_points = 380
        self.s = 2.39996323  # Silver ratio - creates beautiful curlicue pattern

    def update(self):
        self.angle_xz += 0.011  # Smooth rotation
        self.angle_yz += 0.004  # Slow tilt
        self.t += 0.006  # Spiral evolution
        self.hue_offset += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.35
            edges = []
            d = 4.0  # Viewing distance for perspective

            # Generate 3D points using Curlicue formula
            points_3d = []
            cumulative_angle = 0.0
            breath = 1 + 0.15 * math.sin(self.t * 0.8)  # Breathing effect

            for n in range(1, self.n_points + 1):
                # Curlicue: angle accumulates by s*n (creates fractal pattern)
                cumulative_angle += self.s + 0.0001 * math.sin(self.t + n * 0.02)

                # Radius grows with sqrt for even distribution
                r = math.sqrt(n) * 0.08 * breath

                # 3D coordinates - spiral rises in Z
                x = r * math.cos(cumulative_angle)
                y = r * math.sin(cumulative_angle)
                z = (n / self.n_points - 0.5) * 2.5  # Z from -1.25 to 1.25

                points_3d.append((x, y, z, n / self.n_points))

            # Apply 3D rotations and project
            projected = []
            cos_xz, sin_xz = math.cos(self.angle_xz), math.sin(self.angle_xz)
            cos_yz, sin_yz = math.cos(self.angle_yz), math.sin(self.angle_yz)

            for x, y, z, t in points_3d:
                # XZ rotation (around Y axis)
                x2 = x * cos_xz - z * sin_xz
                z2 = x * sin_xz + z * cos_xz

                # YZ rotation (tilt)
                y2 = y * cos_yz - z2 * sin_yz
                z3 = y * sin_yz + z2 * cos_yz

                # Perspective projection
                scale = d / (d - z3) if (d - z3) > 0.1 else d / 0.1
                screen_x = cx + x2 * scale * size
                screen_y = cy - y2 * scale * size

                # Depth for coloring (0 = far, 1 = close)
                depth = (z3 + 2) / 4  # Normalize z3 to 0-1 range

                projected.append((screen_x, screen_y, depth, scale, t))

            # Draw lines connecting consecutive points
            for i in range(1, len(projected)):
                x1, y1, d1, s1, t1 = projected[i - 1]
                x2, y2, d2, s2, t2 = projected[i]

                # Color based on position + depth
                hue = (t2 + self.hue_offset) % 1.0
                avg_depth = (d1 + d2) / 2
                brightness = 0.4 + avg_depth * 0.55  # Closer = brighter
                saturation = 0.7 + avg_depth * 0.25

                # Line width based on scale (perspective)
                avg_scale = (s1 + s2) / 2
                lw = max(1, min(4, int(avg_scale * 1.8)))

                edges.append(
                    (x1, y1, x2, y2, hsv_to_hex(hue, saturation, brightness), lw)
                )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 15: Fourier Epicycles
# =============================================================================
class FourierEpicycles:
    """Epicycles drawing complex shapes - Fourier transform visualization"""

    name = "Fourier Epicycles"
    num_lines = 650  # circles(8*24=192) + arms(8) + trail(400) + buffer

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.trail = []
        self.max_trail = 400
        # Define epicycles: (radius, frequency, phase)
        self.n_cycles = random.randint(5, 8)
        self.cycles = []
        for i in range(self.n_cycles):
            r = 1.0 / (2 * i + 1)  # Decreasing radii
            freq = 2 * i + 1  # Odd harmonics (square wave approx)
            phase = random.random() * math.pi * 2
            self.cycles.append((r * 0.8, freq, phase))

    def update(self):
        self.t += 0.015
        self.hue_offset += 0.003

        # Calculate endpoint (same logic as in get_edges for arm endpoint)
        x, y = 0.0, 0.0
        for r, freq, phase in self.cycles:
            x += r * math.cos(freq * self.t + phase)
            y += r * math.sin(freq * self.t + phase)
        self.trail.append((x, y))
        if len(self.trail) > self.max_trail:
            self.trail = self.trail[-self.max_trail :]

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.3
            edges = []

            # Draw epicycle circles and arms
            x, y = 0.0, 0.0
            for ci, (r, freq, phase) in enumerate(self.cycles):
                # Circle (fewer segments for efficiency)
                n_seg = 16
                hue = (ci / self.n_cycles + self.hue_offset) % 1.0
                for i in range(n_seg):
                    a1 = i * 2 * math.pi / n_seg
                    a2 = (i + 1) * 2 * math.pi / n_seg
                    x1 = cx + (x + r * math.cos(a1)) * size
                    y1 = cy - (y + r * math.sin(a1)) * size
                    x2 = cx + (x + r * math.cos(a2)) * size
                    y2 = cy - (y + r * math.sin(a2)) * size
                    edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.5, 0.4), 1))

                # Arm to next center
                nx = x + r * math.cos(freq * self.t + phase)
                ny = y + r * math.sin(freq * self.t + phase)
                edges.append(
                    (
                        cx + x * size,
                        cy - y * size,
                        cx + nx * size,
                        cy - ny * size,
                        hsv_to_hex(hue, 0.9, 0.85),
                        2,
                    )
                )
                x, y = nx, ny

            # Connect final pendulum position to trail start (if trail exists)
            if len(self.trail) > 0:
                tx, ty = self.trail[-1]
                edges.append(
                    (
                        cx + x * size,
                        cy - y * size,
                        cx + tx * size,
                        cy - ty * size,
                        hsv_to_hex((self.hue_offset + 0.5) % 1.0, 0.95, 0.9),
                        3,
                    )
                )

            # Draw trail
            for i in range(1, len(self.trail)):
                t = i / len(self.trail)
                hue = (t + self.hue_offset + 0.5) % 1.0
                x1, y1 = self.trail[i - 1]
                x2, y2 = self.trail[i]
                edges.append(
                    (
                        cx + x1 * size,
                        cy - y1 * size,
                        cx + x2 * size,
                        cy - y2 * size,
                        hsv_to_hex(hue, 0.9, 0.5 + t * 0.5),
                        3,
                    )
                )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 16: Geodesic Sphere (Triangulated)
# =============================================================================
class GeodesicSphere:
    """Slowly rotating sphere made of triangles - icosahedron subdivision"""

    name = "Geodesic Sphere"
    num_lines = 400

    def __init__(self):
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.hue_offset = 0.0
        self.subdivisions = 2  # Level of detail

        # Generate icosahedron vertices
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.base_verts = [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ]
        # Normalize to unit sphere
        self.base_verts = [self._normalize(v) for v in self.base_verts]

        # Icosahedron faces (20 triangles)
        self.base_faces = [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ]

        # Subdivide for smoother sphere
        self.vertices, self.edges = self._subdivide_icosahedron()

    def _normalize(self, v):
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if length < 0.0001:
            return (0, 0, 1)
        return (v[0] / length, v[1] / length, v[2] / length)

    def _midpoint(self, v1, v2):
        mid = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2, (v1[2] + v2[2]) / 2)
        return self._normalize(mid)  # Project to sphere

    def _subdivide_icosahedron(self):
        vertices = list(self.base_verts)
        faces = list(self.base_faces)

        for _ in range(self.subdivisions):
            new_faces = []
            edge_cache = {}

            for f in faces:
                v0, v1, v2 = f
                # Get or create midpoints
                mids = []
                for edge in [(v0, v1), (v1, v2), (v2, v0)]:
                    key = tuple(sorted(edge))
                    if key not in edge_cache:
                        mid = self._midpoint(vertices[edge[0]], vertices[edge[1]])
                        edge_cache[key] = len(vertices)
                        vertices.append(mid)
                    mids.append(edge_cache[key])

                m01, m12, m20 = mids
                new_faces.extend(
                    [(v0, m01, m20), (v1, m12, m01), (v2, m20, m12), (m01, m12, m20)]
                )
            faces = new_faces

        # Extract unique edges from faces
        edge_set = set()
        for f in faces:
            v0, v1, v2 = f
            for e in [(v0, v1), (v1, v2), (v2, v0)]:
                edge_set.add(tuple(sorted(e)))

        return vertices, list(edge_set)

    def update(self):
        self.angle_x += 0.004  # Very slow rotation
        self.angle_y += 0.006
        self.angle_z += 0.002
        self.hue_offset += 0.002

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.38
            d = 4.0  # Viewing distance
            edges = []

            # Rotation matrices
            cx_r, sx_r = math.cos(self.angle_x), math.sin(self.angle_x)
            cy_r, sy_r = math.cos(self.angle_y), math.sin(self.angle_y)
            cz_r, sz_r = math.cos(self.angle_z), math.sin(self.angle_z)

            # Transform and project vertices
            projected = []
            for vx, vy, vz in self.vertices:
                # Rotate X
                y1 = vy * cx_r - vz * sx_r
                z1 = vy * sx_r + vz * cx_r
                # Rotate Y
                x2 = vx * cy_r + z1 * sy_r
                z2 = -vx * sy_r + z1 * cy_r
                # Rotate Z
                x3 = x2 * cz_r - y1 * sz_r
                y3 = x2 * sz_r + y1 * cz_r

                # Perspective
                scale = d / (d - z2) if (d - z2) > 0.1 else d / 0.1
                sx = cx + x3 * scale * size
                sy = cy - y3 * scale * size
                depth = (z2 + 1.5) / 3  # Normalize depth 0-1

                projected.append((sx, sy, depth, scale))

            # Draw edges
            for i, (v1, v2) in enumerate(self.edges):
                x1, y1, d1, s1 = projected[v1]
                x2, y2, d2, s2 = projected[v2]

                avg_depth = (d1 + d2) / 2
                hue = (i / len(self.edges) + self.hue_offset) % 1.0
                brightness = 0.35 + avg_depth * 0.6
                saturation = 0.6 + avg_depth * 0.35

                lw = max(1, int((s1 + s2) / 2 * 1.5))

                edges.append(
                    (x1, y1, x2, y2, hsv_to_hex(hue, saturation, brightness), lw)
                )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 17: Bouncing Particles
# =============================================================================
class BouncingParticles:
    """Particles bouncing off walls and obstacles - dynamic simulation"""

    name = "Bouncing Particles"
    num_lines = 500

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.n_particles = 35
        self.trail_len = 18
        # Each particle: [x, y, vx, vy, hue, trail]
        self.particles = []
        for i in range(self.n_particles):
            angle = random.random() * 2 * math.pi
            speed = random.uniform(3, 7)
            self.particles.append(
                {
                    "x": random.uniform(0.2, 0.8),
                    "y": random.uniform(0.2, 0.8),
                    "vx": math.cos(angle) * speed * 0.01,
                    "vy": math.sin(angle) * speed * 0.01,
                    "hue": i / self.n_particles,
                    "trail": [],
                }
            )
        self.obstacle_radius = 0.12
        self.obstacle_angle = 0.0

    def update(self):
        self.t += 0.016
        self.hue_offset += 0.002
        self.obstacle_angle += 0.008

        # Update particles
        for p in self.particles:
            # Store trail
            p["trail"].append((p["x"], p["y"]))
            if len(p["trail"]) > self.trail_len:
                p["trail"] = p["trail"][-self.trail_len :]

            # Move
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Bounce off walls
            if p["x"] < 0.08 or p["x"] > 0.92:
                p["vx"] *= -1
                p["x"] = max(0.08, min(0.92, p["x"]))
                p["hue"] = (p["hue"] + 0.1) % 1.0
            if p["y"] < 0.12 or p["y"] > 0.88:
                p["vy"] *= -1
                p["y"] = max(0.12, min(0.88, p["y"]))
                p["hue"] = (p["hue"] + 0.1) % 1.0

            # Bounce off central obstacle (moving)
            obs_x = 0.5 + 0.15 * math.cos(self.obstacle_angle)
            obs_y = 0.5 + 0.15 * math.sin(self.obstacle_angle * 0.7)
            dx = p["x"] - obs_x
            dy = p["y"] - obs_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.obstacle_radius and dist > 0.001:
                # Normalize and reflect
                nx, ny = dx / dist, dy / dist
                dot = p["vx"] * nx + p["vy"] * ny
                if dot < 0:  # Moving towards obstacle
                    p["vx"] -= 2 * dot * nx
                    p["vy"] -= 2 * dot * ny
                    # Push out
                    p["x"] = obs_x + nx * (self.obstacle_radius + 0.01)
                    p["y"] = obs_y + ny * (self.obstacle_radius + 0.01)
                    p["hue"] = (p["hue"] + 0.15) % 1.0

    def get_edges(self, width, height):
        try:
            edges = []

            # Draw boundary rectangle
            margin = min(width, height) * 0.08
            corners = [
                (margin, margin * 1.5),
                (width - margin, margin * 1.5),
                (width - margin, height - margin),
                (margin, height - margin),
            ]
            for i in range(4):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % 4]
                hue = (i * 0.25 + self.hue_offset) % 1.0
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.4, 0.4), 2))

            # Draw moving obstacle circle
            obs_x = 0.5 + 0.15 * math.cos(self.obstacle_angle)
            obs_y = 0.5 + 0.15 * math.sin(self.obstacle_angle * 0.7)
            obs_px = margin + obs_x * (width - 2 * margin)
            obs_py = margin * 1.5 + obs_y * (height - margin * 2.5)
            obs_r = self.obstacle_radius * min(
                width - 2 * margin, height - margin * 2.5
            )
            n_seg = 24
            for i in range(n_seg):
                a1 = i * 2 * math.pi / n_seg
                a2 = (i + 1) * 2 * math.pi / n_seg
                x1 = obs_px + obs_r * math.cos(a1)
                y1 = obs_py + obs_r * math.sin(a1)
                x2 = obs_px + obs_r * math.cos(a2)
                y2 = obs_py + obs_r * math.sin(a2)
                hue = (i / n_seg + self.hue_offset + 0.5) % 1.0
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, 0.6, 0.55), 2))

            # Draw particles and trails
            w_inner = width - 2 * margin
            h_inner = height - margin * 2.5
            for p in self.particles:
                trail = p["trail"]
                hue = (p["hue"] + self.hue_offset) % 1.0

                # Draw trail
                for i in range(1, len(trail)):
                    t = i / len(trail)
                    x1 = margin + trail[i - 1][0] * w_inner
                    y1 = margin * 1.5 + trail[i - 1][1] * h_inner
                    x2 = margin + trail[i][0] * w_inner
                    y2 = margin * 1.5 + trail[i][1] * h_inner
                    trail_hue = (hue + (1 - t) * 0.2) % 1.0
                    edges.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            hsv_to_hex(trail_hue, 0.85, 0.3 + t * 0.6),
                            max(1, int(t * 3)),
                        )
                    )

                # Draw current position as small cross
                px = margin + p["x"] * w_inner
                py = margin * 1.5 + p["y"] * h_inner
                size = 4
                edges.append(
                    (px - size, py, px + size, py, hsv_to_hex(hue, 0.9, 0.95), 2)
                )
                edges.append(
                    (px, py - size, px, py + size, hsv_to_hex(hue, 0.9, 0.95), 2)
                )

            return edges
        except Exception:
            return []


# =============================================================================
# VISUALIZATION 18: Tunnel Flight (Curved Warp Tunnel)
# =============================================================================
class TunnelFlight:
    """High-speed flight through a curved tunnel of rectangles"""

    name = "Tunnel Flight"
    num_lines = 500

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.speed = 0.06  # How fast frames approach
        self.n_frames = 7  # Number of rectangular frames in tunnel
        self.curve_freq_x = 0.4  # Curvature frequency X
        self.curve_freq_y = 0.3  # Curvature frequency Y
        self.curve_amp = 0.35  # Curvature amplitude
        # Initialize frame Z positions (0 = closest, 1 = farthest)
        self.frames = [
            {"z": i / self.n_frames, "phase": random.random() * math.pi * 2}
            for i in range(self.n_frames)
        ]

    def update(self):
        self.t += self.speed
        self.hue_offset += 0.004
        # Move frames towards viewer
        for f in self.frames:
            f["z"] -= self.speed * 0.08
            # Respawn at far end when passing viewer
            if f["z"] < 0.02:
                f["z"] = 1.0
                f["phase"] = random.random() * math.pi * 2

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            edges = []

            # Sort frames by Z (far to near) for proper depth rendering
            sorted_frames = sorted(self.frames, key=lambda f: -f["z"])

            for fi, frame in enumerate(sorted_frames):
                z = frame["z"]
                if z < 0.03:
                    continue  # Skip frames too close

                # Perspective scale: closer = larger
                # Using exponential for more dramatic effect
                perspective = 1.0 / (z * 2.5 + 0.1)

                # Curved path offset based on Z depth
                # This creates the winding tunnel effect
                curve_x = self.curve_amp * math.sin(self.t * self.curve_freq_x + z * 8)
                curve_y = self.curve_amp * math.cos(
                    self.t * self.curve_freq_y + z * 6 + 1.5
                )

                # Frame center position (offset by curve)
                frame_cx = cx + curve_x * width * 0.4 * (1 - z)
                frame_cy = cy + curve_y * height * 0.3 * (1 - z)

                # Rectangle size (larger when closer)
                base_size = min(width, height) * 0.45
                rect_w = base_size * perspective
                rect_h = base_size * perspective * 0.7  # Slightly shorter height

                # Slight rotation for added dynamism
                rotation = math.sin(self.t * 0.3 + z * 4) * 0.15
                cos_r, sin_r = math.cos(rotation), math.sin(rotation)

                # Four corners of rectangle (before rotation)
                corners_local = [
                    (-rect_w / 2, -rect_h / 2),
                    (rect_w / 2, -rect_h / 2),
                    (rect_w / 2, rect_h / 2),
                    (-rect_w / 2, rect_h / 2),
                ]

                # Apply rotation and translate to frame center
                corners = []
                for lx, ly in corners_local:
                    rx = lx * cos_r - ly * sin_r
                    ry = lx * sin_r + ly * cos_r
                    corners.append((frame_cx + rx, frame_cy + ry))

                # Color based on depth (far = darker/bluer, close = brighter)
                depth_factor = 1 - z  # 0 when far, 1 when close
                hue = (z * 0.6 + self.hue_offset) % 1.0
                saturation = 0.6 + depth_factor * 0.35
                brightness = 0.25 + depth_factor * 0.7

                color = hsv_to_hex(hue, saturation, brightness)
                lw = max(1, int(2 + depth_factor * 3))

                # Draw rectangle edges
                for i in range(4):
                    x1, y1 = corners[i]
                    x2, y2 = corners[(i + 1) % 4]
                    edges.append((x1, y1, x2, y2, color, lw))

                # Add inner detail lines for closer frames
                if z < 0.5 and depth_factor > 0.3:
                    inner_scale = 0.7
                    inner_corners = []
                    for lx, ly in corners_local:
                        rx = lx * inner_scale * cos_r - ly * inner_scale * sin_r
                        ry = lx * inner_scale * sin_r + ly * inner_scale * cos_r
                        inner_corners.append((frame_cx + rx, frame_cy + ry))

                    inner_hue = (hue + 0.15) % 1.0
                    inner_color = hsv_to_hex(
                        inner_hue, saturation * 0.8, brightness * 0.7
                    )
                    inner_lw = max(1, lw - 1)

                    for i in range(4):
                        x1, y1 = inner_corners[i]
                        x2, y2 = inner_corners[(i + 1) % 4]
                        edges.append((x1, y1, x2, y2, inner_color, inner_lw))

                    # Connecting lines from outer to inner corners
                    for i in range(4):
                        ox, oy = corners[i]
                        ix, iy = inner_corners[i]
                        conn_color = hsv_to_hex(
                            (hue + 0.3) % 1.0, saturation * 0.6, brightness * 0.5
                        )
                        edges.append((ox, oy, ix, iy, conn_color, 1))

            # Add motion blur / speed lines at edges
            n_speed_lines = 12
            for i in range(n_speed_lines):
                angle = i * 2 * math.pi / n_speed_lines
                # Lines emanate from center towards edges
                inner_r = min(width, height) * 0.1
                outer_r = min(width, height) * 0.48

                # Animate the speed lines
                phase = self.t * 3 + i * 0.5
                line_alpha = 0.3 + 0.3 * math.sin(phase)

                x1 = cx + inner_r * math.cos(angle)
                y1 = cy + inner_r * math.sin(angle)
                x2 = cx + outer_r * math.cos(angle)
                y2 = cy + outer_r * math.sin(angle)

                speed_hue = (i / n_speed_lines + self.hue_offset + 0.5) % 1.0
                speed_color = hsv_to_hex(speed_hue, 0.5, line_alpha)
                edges.append((x1, y1, x2, y2, speed_color, 1))

            return edges
        except Exception:
            return []


# =============================================================================
# CURSOR TRACKER
# =============================================================================


class CursorTracker:
    """Tracks mouse position, detects cursor absence after timeout"""

    def __init__(self):
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.last_move_time = 0.0
        self.cursor_present = False

    def on_motion(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.last_move_time = time.time()
        self.cursor_present = True

    def update(self):
        if time.time() - self.last_move_time > CONFIG["cursor_timeout"]:
            self.cursor_present = False


# =============================================================================
# WANDERING BODY (BASE CLASS)
# =============================================================================


class WanderingBody:
    """Base class for cursor-following creatures with wandering fallback"""

    def __init__(self, x, y, speed=1.0):
        self.x = x
        self.y = y
        self.speed = speed
        self.target_x = x
        self.target_y = y
        self.wander_timer = 0
        self.phase = random.random() * math.pi * 2  # for organic drift
        self.width = 900  # updated on resize
        self.height = 700

    def update_position(self, cursor):
        """Update position based on cursor state"""
        self.phase += 0.02

        if cursor.cursor_present:
            dx = cursor.mouse_x - self.x
            dy = cursor.mouse_y - self.y
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            if dist > CONFIG["min_follow_distance"]:
                self.x += dx * CONFIG["follow_ease"] * self.speed
                self.y += dy * CONFIG["follow_ease"] * self.speed
            # Perpendicular organic drift
            perp_x = -dy / dist * math.sin(self.phase) * 0.5
            perp_y = dx / dist * math.sin(self.phase) * 0.5
            self.x += perp_x
            self.y += perp_y
        else:
            # Wander mode
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 50:
                # Pick new random target
                margin = 100
                self.target_x = random.uniform(margin, self.width - margin)
                self.target_y = random.uniform(margin, self.height - margin)
            self.x += dx * CONFIG["wander_ease"] * self.speed
            self.y += dy * CONFIG["wander_ease"] * self.speed

    def resize(self, w, h):
        self.width = w
        self.height = h

    def get_edges(self):
        """Override in subclass — returns list of (x1,y1,x2,y2,color,linewidth)"""
        return []


# =============================================================================
# HILBERT CURVE
# =============================================================================


class HilbertCurve:
    """Order-4 Hilbert curve — morphs from a straight line into the full fractal."""

    name = "Hilbert Curve"
    num_lines = 255

    def __init__(self):
        self.order = CONFIG["hilbert_order"]
        self.n = 1 << self.order  # grid size: 16 for order=4
        self.total = self.n * self.n  # 256 points
        self.hue_offset = 0.0
        self.morph_time = 0.0
        # Line positions: all 256 points evenly spaced on a horizontal line
        self._line_points = [(i / (self.total - 1), 0.5) for i in range(self.total)]
        # Curve positions: normalized Hilbert curve in [0,1]x[0,1]
        self._curve_points = self._build_normalized_points()

    # ------------------------------------------------------------------
    # Hilbert index → (x, y) coordinate
    # ------------------------------------------------------------------
    def _hilbert_d2xy(self, n, d):
        """Convert Hilbert curve index d to (x,y) for n×n grid"""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if ((d & 1) ^ rx) else 0  # XOR
            # Rotate quadrant
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    def _build_normalized_points(self):
        """Build list of (nx, ny) in [0..1] x [0..1] for all curve indices."""
        n = self.n
        pts = []
        for d in range(self.total):
            gx, gy = self._hilbert_d2xy(n, d)
            pts.append((gx / (n - 1), gy / (n - 1)))
        return pts

    # ------------------------------------------------------------------
    # Quadrant-based hue mapping
    # ------------------------------------------------------------------
    def _segment_color(self, x_norm, y_norm, t, seg_frac):
        """
        Return HSV hue for a segment at normalized canvas position.
        t          = position within quadrant [0..1]
        seg_frac   = overall fraction along entire curve [0..1]
        """
        in_left = x_norm < 0.5
        in_top = y_norm < 0.5

        if in_top and in_left:  # Q1: blue → cyan
            hue_base = 0.55 + t * 0.20
        elif in_top and not in_left:  # Q2: green → yellow
            hue_base = 0.20 + t * 0.20
        elif not in_top and in_left:  # Q3: yellow → orange
            hue_base = 0.08 + t * 0.10
        else:  # Q4: pink → magenta
            hue_base = 0.80 + t * 0.15

        hue = (hue_base + self.hue_offset) % 1.0
        # Slightly vary brightness by position for depth
        value = 0.65 + 0.25 * math.sin(seg_frac * math.pi * 4 + self.hue_offset * 8)
        saturation = 0.75 + 0.20 * math.cos(seg_frac * math.pi * 6)
        saturation = max(0.0, min(1.0, saturation))
        value = max(0.0, min(1.0, value))
        return hsv_to_hex(hue, saturation, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self):
        self.hue_offset = (self.hue_offset + CONFIG["color_cycle_speed"]) % 1.0
        self.morph_time += 0.02

    def _morph_factor(self):
        """Returns 0.0 (line) to 1.0 (curve), cycling over 9 seconds."""
        t = self.morph_time % 9.0
        if t < 3.0:  # line → curve (ease-out)
            g = t / 3.0
            return g * (2 - g)
        elif t < 5.0:  # hold as curve
            return 1.0
        elif t < 8.0:  # curve → line (ease-in)
            g = (t - 5.0) / 3.0
            return 1.0 - g * g
        else:  # brief pause as line
            return 0.0

    def get_edges(self, width, height):
        """Hilbert curve morphing from a straight horizontal line into the fractal."""
        margin = 60
        side = min(width, height) - 2 * margin
        if side < 10:
            return []

        cx, cy = width / 2, height / 2
        x0, y0 = cx - side / 2, cy - side / 2
        morph = self._morph_factor()
        total = self.total
        lw = 1.5 + morph * 1.0
        edges = []

        for i in range(total - 1):
            lx1, ly1 = self._line_points[i]
            cpx1, cpy1 = self._curve_points[i]
            nx1 = lx1 + (cpx1 - lx1) * morph
            ny1 = ly1 + (cpy1 - ly1) * morph

            lx2, ly2 = self._line_points[i + 1]
            cpx2, cpy2 = self._curve_points[i + 1]
            nx2 = lx2 + (cpx2 - lx2) * morph
            ny2 = ly2 + (cpy2 - ly2) * morph

            sx1 = x0 + nx1 * side
            sy1 = y0 + ny1 * side
            sx2 = x0 + nx2 * side
            sy2 = y0 + ny2 * side

            # Coloring based on CURVE positions (stable during morph)
            mx_norm = (cpx1 + cpx2) / 2
            my_norm = (cpy1 + cpy2) / 2
            t_qx = (mx_norm % 0.5) / 0.5
            t_qy = (my_norm % 0.5) / 0.5
            t = (t_qx + t_qy) / 2
            seg_frac = i / (total - 1)

            color = self._segment_color(mx_norm, my_norm, t, seg_frac)
            edges.append((sx1, sy1, sx2, sy2, color, max(1, lw)))

        return edges


# =============================================================================
# IK HELPER (shared by SpiderCreature + StickPerson)
# =============================================================================


def solve_2bone_ik(hip_x, hip_y, foot_x, foot_y, seg1_len, seg2_len, side=1):
    """
    2-segment IK: returns knee (x, y).
    side=1 → knee bends to the left of hip→foot direction
    side=-1 → knee bends to the right
    """
    dx = foot_x - hip_x
    dy = foot_y - hip_y
    dist = math.sqrt(dx * dx + dy * dy) + 0.0001
    # Clamp so the legs don't over-extend (triangle inequality)
    dist = min(dist, seg1_len + seg2_len - 0.5)
    dist = max(dist, abs(seg1_len - seg2_len) + 0.5)

    cos_angle = (seg1_len**2 + dist**2 - seg2_len**2) / (2 * seg1_len * dist)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_to_target = math.atan2(dy, dx)
    angle_offset = math.acos(cos_angle)

    knee_angle = angle_to_target - angle_offset * side
    knee_x = hip_x + seg1_len * math.cos(knee_angle)
    knee_y = hip_y + seg1_len * math.sin(knee_angle)
    return knee_x, knee_y


# =============================================================================
# CREATURE 1: SpiderCreature — 8-legged beauty with tetrapod gait + IK
# =============================================================================


class SpiderCreature(WanderingBody):
    """
    8-legged spider with:
    - 2-segment IK per leg
    - Tetrapod alternating gait (groups A & B)
    - Organic foot stepping with parabolic lift arc
    - Warm color cycling (red → orange → yellow)
    - Body segments drawn as crossing diamonds
    """

    def __init__(self, x, y):
        super().__init__(x, y, speed=1.2)
        self.hue_offset = 0.0
        self.base_hue = 0.02  # deep red-orange
        self.num_legs = 8
        self.seg1_len = 52
        self.seg2_len = 50
        self.body_radius = 22

        # Leg attachment angles (relative to body center, in radians)
        spread = math.pi / 5.5  # ~33° between pairs
        self.hip_angles = []
        for i in range(4):
            angle_right = -spread * (1.5 - i)
            angle_left = math.pi + spread * (i - 1.5)
            self.hip_angles.append(angle_right)
            self.hip_angles.append(angle_left)

        self.hip_offset = self.body_radius * 0.9

        # Ground targets for each foot (world coords)
        self.foot_ground = []
        self.foot_current = []
        self.stepping = []
        self.step_progress = []
        self.step_from = []
        self.step_to = []
        self.step_height = []

        # Leg IK side — which way knee bends
        self.ik_side = [1, -1, 1, -1, 1, -1, 1, -1]

        # Initialize foot positions at natural rest stance
        for i in range(self.num_legs):
            angle = self.hip_angles[i]
            rest_dist = (self.seg1_len + self.seg2_len) * 0.7
            gx = x + math.cos(angle) * rest_dist
            gy = y + math.sin(angle) * rest_dist
            self.foot_ground.append([gx, gy])
            self.foot_current.append([gx, gy])
            self.stepping.append(False)
            self.step_progress.append(0.0)
            self.step_from.append([gx, gy])
            self.step_to.append([gx, gy])
            self.step_height.append(18.0)

        # Gait cooldowns
        self.gait_cooldown = [0] * self.num_legs

        # Body bob animation
        self.body_bob = 0.0
        self.prev_x = x
        self.prev_y = y
        self.velocity = 0.0

        # Eye blink state
        self.blink_timer = 0
        self.eye_open = 1.0

    def _get_hip_pos(self, leg_idx):
        """Hip socket position in world space."""
        angle = self.hip_angles[leg_idx]
        hx = self.x + math.cos(angle) * self.hip_offset
        hy = self.y + math.sin(angle) * self.hip_offset
        return hx, hy

    def _ideal_foot_target(self, leg_idx):
        """Where this foot 'wants' to be when stepping."""
        angle = self.hip_angles[leg_idx]
        reach = self.seg1_len + self.seg2_len
        vx = self.x - self.prev_x
        vy = self.y - self.prev_y
        spd = math.sqrt(vx * vx + vy * vy) + 0.001
        overshoot = min(spd * 8, 25)
        tx = self.x + math.cos(angle) * (reach * 0.68) + vx / spd * overshoot
        ty = self.y + math.sin(angle) * (reach * 0.68) + vy / spd * overshoot
        return tx, ty

    def _dist_from_home(self, leg_idx):
        """How far current ground target is from ideal position."""
        gx, gy = self.foot_ground[leg_idx]
        tx, ty = self._ideal_foot_target(leg_idx)
        return math.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)

    def update_animation(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        self.velocity = math.sqrt(dx * dx + dy * dy)
        self.prev_x = self.x
        self.prev_y = self.y

        self.body_bob = math.sin(self.phase * 3) * 1.5
        self.hue_offset = (self.hue_offset + 0.003) % 1.0

        # Eye blink
        self.blink_timer += 1
        if self.blink_timer > 120:
            self.eye_open = max(0.0, self.eye_open - 0.15)
            if self.eye_open <= 0.0:
                self.blink_timer = 0
                self.eye_open = 1.0

        # Update stepping legs
        step_thresh = max(22, min(45, self.velocity * 20 + 22))

        for i in range(self.num_legs):
            self.gait_cooldown[i] += 1

            if self.stepping[i]:
                self.step_progress[i] += 0.13
                t = min(self.step_progress[i], 1.0)
                t_ease = t * t * (3 - 2 * t)
                fx = self.step_from[i][0] * (1 - t_ease) + self.step_to[i][0] * t_ease
                fy = self.step_from[i][1] * (1 - t_ease) + self.step_to[i][1] * t_ease
                lift = math.sin(t * math.pi) * self.step_height[i]
                self.foot_current[i][0] = fx
                self.foot_current[i][1] = fy - lift

                if t >= 1.0:
                    self.stepping[i] = False
                    self.foot_ground[i][0] = self.step_to[i][0]
                    self.foot_ground[i][1] = self.step_to[i][1]
                    self.foot_current[i][0] = self.step_to[i][0]
                    self.foot_current[i][1] = self.step_to[i][1]
            else:
                dist_off = self._dist_from_home(i)
                own_stepping = sum(
                    1
                    for j in range(self.num_legs)
                    if (j % 2 == i % 2) and self.stepping[j]
                )
                can_step = (
                    dist_off > step_thresh
                    and own_stepping < 2
                    and self.gait_cooldown[i] > 6
                )
                if can_step:
                    self.stepping[i] = True
                    self.step_progress[i] = 0.0
                    self.step_from[i] = [
                        self.foot_current[i][0],
                        self.foot_current[i][1],
                    ]
                    tx, ty = self._ideal_foot_target(i)
                    self.step_to[i] = [tx, ty]
                    self.step_height[i] = 14 + self.velocity * 12
                    self.gait_cooldown[i] = 0

    def get_edges(self):
        edges = []
        by = self.y + self.body_bob

        # --- Body: layered diamond / cross ---
        bh = self.base_hue + self.hue_offset
        body_color = hsv_to_hex(bh % 1.0, 0.85, 0.88)
        body_color2 = hsv_to_hex((bh + 0.06) % 1.0, 0.9, 0.75)
        body_color3 = hsv_to_hex((bh + 0.12) % 1.0, 0.7, 0.95)

        br = self.body_radius
        bx = self.x
        # Outer diamond (abdomen)
        pts = [
            (bx, by - br * 1.2),
            (bx + br, by),
            (bx, by + br * 0.8),
            (bx - br, by),
        ]
        for k in range(4):
            x1, y1 = pts[k]
            x2, y2 = pts[(k + 1) % 4]
            edges.append((x1, y1, x2, y2, body_color, 2.5))

        # Inner cross
        inner = br * 0.55
        edges.append(
            (
                bx - inner,
                by - inner * 0.7,
                bx + inner,
                by + inner * 0.7,
                body_color2,
                1.5,
            )
        )
        edges.append(
            (
                bx + inner,
                by - inner * 0.7,
                bx - inner,
                by + inner * 0.7,
                body_color2,
                1.5,
            )
        )

        # Cephalothorax (front lobe)
        ct_offset = -br * 0.72
        ct_r = br * 0.52
        ct_pts = [
            (bx, by + ct_offset - ct_r * 0.9),
            (bx + ct_r * 0.8, by + ct_offset),
            (bx, by + ct_offset + ct_r * 0.6),
            (bx - ct_r * 0.8, by + ct_offset),
        ]
        for k in range(4):
            x1, y1 = ct_pts[k]
            x2, y2 = ct_pts[(k + 1) % 4]
            edges.append((x1, y1, x2, y2, body_color3, 2.0))

        # Eyes
        eye_spread = ct_r * 0.35
        ey = by + ct_offset - ct_r * 0.1
        eye_size = 3.5 * self.eye_open
        if eye_size > 0.5:
            eye_color = hsv_to_hex((bh + 0.15) % 1.0, 0.5, 1.0)
            for ex_off in [-eye_spread, eye_spread]:
                ex = bx + ex_off
                edges.append((ex - eye_size, ey, ex + eye_size, ey, eye_color, 1.5))
                edges.append(
                    (
                        ex,
                        ey - eye_size * self.eye_open,
                        ex,
                        ey + eye_size * self.eye_open,
                        eye_color,
                        1.5,
                    )
                )

        # --- Legs ---
        for i in range(self.num_legs):
            hx, hy = self._get_hip_pos(i)
            hy += self.body_bob

            fx = self.foot_current[i][0]
            fy = self.foot_current[i][1]

            kx, ky = solve_2bone_ik(
                hx, hy, fx, fy, self.seg1_len, self.seg2_len, self.ik_side[i]
            )

            leg_hue = (bh + i * 0.035) % 1.0
            sat = 0.82 + math.sin(self.phase + i) * 0.06
            val = 0.78 + math.cos(self.phase * 0.7 + i * 0.5) * 0.08

            lw_upper = max(1.5, 2.8 - i * 0.12)
            lw_lower = max(1.0, 1.8 - i * 0.08)

            seg_color = hsv_to_hex(leg_hue, sat, val)
            tip_color = hsv_to_hex((leg_hue + 0.08) % 1.0, sat * 0.9, val * 0.85)

            edges.append((hx, hy, kx, ky, seg_color, lw_upper))
            edges.append((kx, ky, fx, fy, tip_color, lw_lower))

            # Foot claw
            claw_len = 6
            foot_angle = math.atan2(fy - ky, fx - kx)
            for claw_off in [-0.35, 0.35]:
                ca = foot_angle + claw_off
                cx2 = fx + math.cos(ca) * claw_len
                cy2 = fy + math.sin(ca) * claw_len
                edges.append((fx, fy, cx2, cy2, tip_color, 1.0))

        return edges


# =============================================================================
# CREATURE 2: StickPerson — bipedal walking figure with IK + facing
# =============================================================================


class StickPerson(WanderingBody):
    """
    Stick figure with:
    - Procedural walk cycle (sine-wave limb swing)
    - Direction-aware facing (rotates with movement)
    - 2-segment IK for legs and arms
    - Idle sway when stationary
    - Cool color palette (cyan → blue → purple)
    """

    def __init__(self, x, y):
        super().__init__(x, y, speed=0.8)
        self.hue_offset = 0.0
        self.base_hue = 0.55  # cyan-blue
        self.walk_phase = 0.0
        self.facing_angle = 0.0
        self.target_facing = 0.0
        self.prev_x = x
        self.prev_y = y
        self.velocity = 0.0
        self.is_moving = False

        # Body proportions
        self.head_radius = 12
        self.torso_length = 42
        self.upper_arm = 19
        self.forearm = 17
        self.thigh = 26
        self.shin = 24

        # Foot IK state (left, right)
        self.foot_ground = [
            [x - 8, y + self.torso_length + self.thigh + self.shin],
            [x + 8, y + self.torso_length + self.thigh + self.shin],
        ]
        self.foot_current = [
            [x - 8, y + self.torso_length + self.thigh + self.shin],
            [x + 8, y + self.torso_length + self.thigh + self.shin],
        ]
        self.foot_stepping = [False, False]
        self.foot_step_progress = [0.0, 0.0]
        self.foot_step_from = [list(self.foot_ground[0]), list(self.foot_ground[1])]
        self.foot_step_to = [list(self.foot_ground[0]), list(self.foot_ground[1])]
        self.last_step_leg = 0
        self.step_cooldown = 0

        self.body_bob = 0.0
        self.breath_phase = random.random() * math.pi * 2
        self.head_verts = 10

    def _rotate_point(self, px, py, cx, cy, angle):
        dx, dy = px - cx, py - cy
        c, s = math.cos(angle), math.sin(angle)
        return cx + dx * c - dy * s, cy + dx * s + dy * c

    def update_animation(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        self.velocity = math.sqrt(dx * dx + dy * dy)
        self.is_moving = self.velocity > 0.3

        if self.is_moving:
            self.target_facing = math.atan2(dy, dx)

        # Smooth facing rotation
        angle_diff = self.target_facing - self.facing_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        self.facing_angle += angle_diff * 0.08

        self.prev_x = self.x
        self.prev_y = self.y

        # Walk phase
        walk_speed = self.velocity * 2.5 + 0.01
        if self.is_moving:
            self.walk_phase += walk_speed
        else:
            self.walk_phase += 0.015

        self.breath_phase += 0.03

        bob_amp = min(self.velocity * 8, 3.0)
        self.body_bob = math.sin(self.walk_phase * 2) * bob_amp

        self.hue_offset = (self.hue_offset + 0.002) % 1.0

        # Foot IK stepping
        if self.is_moving:
            self.step_cooldown = max(0, self.step_cooldown - 1)
            hip_x, hip_y = self._get_hip_pos()

            for foot_idx in range(2):
                if self.foot_stepping[foot_idx]:
                    self.foot_step_progress[foot_idx] += 0.15
                    t = min(self.foot_step_progress[foot_idx], 1.0)
                    t_ease = t * t * (3 - 2 * t)
                    fx = (
                        self.foot_step_from[foot_idx][0] * (1 - t_ease)
                        + self.foot_step_to[foot_idx][0] * t_ease
                    )
                    fy = (
                        self.foot_step_from[foot_idx][1] * (1 - t_ease)
                        + self.foot_step_to[foot_idx][1] * t_ease
                    )
                    lift = math.sin(t * math.pi) * 16
                    self.foot_current[foot_idx][0] = fx
                    self.foot_current[foot_idx][1] = fy - lift
                    if t >= 1.0:
                        self.foot_stepping[foot_idx] = False
                        self.foot_ground[foot_idx] = [fx, fy]
                        self.foot_current[foot_idx] = [fx, fy]

            # Trigger new step if foot is far from ideal
            if self.step_cooldown == 0:
                for foot_idx in range(2):
                    if (
                        not self.foot_stepping[foot_idx]
                        and not self.foot_stepping[1 - foot_idx]
                    ):
                        ideal_x, ideal_y = self._ideal_foot(foot_idx, hip_x, hip_y)
                        gx, gy = self.foot_ground[foot_idx]
                        dist = math.sqrt((gx - ideal_x) ** 2 + (gy - ideal_y) ** 2)
                        if dist > 20:
                            self.foot_stepping[foot_idx] = True
                            self.foot_step_progress[foot_idx] = 0.0
                            self.foot_step_from[foot_idx] = list(
                                self.foot_current[foot_idx]
                            )
                            self.foot_step_to[foot_idx] = [ideal_x, ideal_y]
                            self.step_cooldown = 3
                            break
        else:
            # Drift feet back toward rest when standing still
            hip_x, hip_y = self._get_hip_pos()
            for foot_idx in range(2):
                rest_x, rest_y = self._rest_foot(foot_idx, hip_x, hip_y)
                self.foot_current[foot_idx][0] += (
                    rest_x - self.foot_current[foot_idx][0]
                ) * 0.04
                self.foot_current[foot_idx][1] += (
                    rest_y - self.foot_current[foot_idx][1]
                ) * 0.04
                self.foot_ground[foot_idx] = list(self.foot_current[foot_idx])

    def _get_hip_pos(self):
        return self.x, self.y + self.torso_length + self.body_bob

    def _get_shoulder_pos(self):
        return self.x, self.y + self.body_bob - 2

    def _rest_foot(self, foot_idx, hip_x, hip_y):
        side = -1 if foot_idx == 0 else 1
        perp = self.facing_angle + math.pi / 2
        fx = hip_x + math.cos(perp) * side * 9
        fy = hip_y + self.thigh + self.shin - 2
        return fx, fy

    def _ideal_foot(self, foot_idx, hip_x, hip_y):
        side = -1 if foot_idx == 0 else 1
        perp = self.facing_angle + math.pi / 2
        forward_offset = self.velocity * 15
        step_x = (
            hip_x
            + math.cos(self.facing_angle) * forward_offset
            + math.cos(perp) * side * 10
        )
        step_y = (
            hip_y
            + self.thigh
            + self.shin
            - 2
            + math.sin(self.facing_angle) * forward_offset * 0.4
        )
        return step_x, step_y

    def _arm_swing(self, arm_idx):
        sx, sy = self._get_shoulder_pos()
        side = -1 if arm_idx == 0 else 1
        perp = self.facing_angle + math.pi / 2

        swing_amp = min(self.velocity * 20 + (0.08 if self.is_moving else 0.04), 0.45)
        phase_offset = 0 if arm_idx == 0 else math.pi
        swing_angle = math.sin(self.walk_phase + phase_offset) * swing_amp

        upper_angle = math.pi / 2 + swing_angle * 0.7

        shoulder_socket_x = sx + math.cos(perp) * side * 8
        shoulder_socket_y = sy + 4

        elbow_x = (
            shoulder_socket_x + math.cos(upper_angle) * self.upper_arm * side * 0.3
        )
        elbow_y = shoulder_socket_y + math.sin(math.pi / 2) * self.upper_arm

        forearm_angle = upper_angle + 0.25
        hand_x = elbow_x + math.cos(forearm_angle) * self.forearm * side * 0.2
        hand_y = elbow_y + self.forearm * 0.95

        return shoulder_socket_x, shoulder_socket_y, elbow_x, elbow_y, hand_x, hand_y

    def get_edges(self):
        edges = []
        bh = self.base_hue + self.hue_offset

        head_color = hsv_to_hex(bh % 1.0, 0.75, 0.85)
        torso_color = hsv_to_hex((bh + 0.05) % 1.0, 0.80, 0.80)
        arm_color = hsv_to_hex((bh + 0.10) % 1.0, 0.82, 0.78)
        forearm_color = hsv_to_hex((bh + 0.13) % 1.0, 0.85, 0.72)
        leg_color = hsv_to_hex((bh + 0.17) % 1.0, 0.88, 0.75)
        shin_color = hsv_to_hex((bh + 0.22) % 1.0, 0.90, 0.68)

        bx = self.x
        by = self.y + self.body_bob

        # --- Head (polygon loop) ---
        hr = self.head_radius
        sway = math.sin(self.breath_phase) * (0.06 if not self.is_moving else 0.03)
        head_cx = bx + math.cos(self.facing_angle) * 2
        head_cy = by - 4

        n = self.head_verts
        head_pts = []
        for k in range(n):
            a = 2 * math.pi * k / n + sway
            hpx = head_cx + math.cos(a) * hr
            hpy = head_cy + math.sin(a) * hr * 0.95
            head_pts.append((hpx, hpy))

        for k in range(n):
            x1, y1 = head_pts[k]
            x2, y2 = head_pts[(k + 1) % n]
            edges.append((x1, y1, x2, y2, head_color, 2.2))

        # Eyes
        eye_offset_fwd = math.cos(self.facing_angle) * 4
        eye_offset_up = -hr * 0.18
        eye_len = 3.5
        eye_color = hsv_to_hex((bh + 0.35) % 1.0, 0.5, 1.0)
        perp = self.facing_angle + math.pi / 2
        for eye_side in [-1, 1]:
            ex = head_cx + eye_offset_fwd + math.cos(perp) * eye_side * 4.5
            ey = head_cy + eye_offset_up
            edges.append((ex - eye_len, ey, ex + eye_len, ey, eye_color, 1.5))

        # --- Torso ---
        torso_top_x = bx
        torso_top_y = by
        hip_x, hip_y = self._get_hip_pos()

        lean = math.cos(self.facing_angle) * self.velocity * 6
        torso_bot_x = bx + lean * 0.3
        torso_bot_y = hip_y

        edges.append(
            (torso_top_x, torso_top_y, torso_bot_x, torso_bot_y, torso_color, 2.8)
        )

        # --- Arms ---
        for arm_idx in range(2):
            sx2, sy2, ex2, ey2, hx2, hy2 = self._arm_swing(arm_idx)
            edges.append((sx2, sy2, ex2, ey2, arm_color, 2.2))
            edges.append((ex2, ey2, hx2, hy2, forearm_color, 1.7))
            hand_sz = 2.5
            edges.append((hx2 - hand_sz, hy2, hx2 + hand_sz, hy2, forearm_color, 1.0))

        # --- Legs (IK) ---
        for leg_idx in range(2):
            fx = self.foot_current[leg_idx][0]
            fy = self.foot_current[leg_idx][1]

            side = -1 if leg_idx == 0 else 1
            kx, ky = solve_2bone_ik(hip_x, hip_y, fx, fy, self.thigh, self.shin, side)

            edges.append((hip_x, hip_y, kx, ky, leg_color, 2.5))
            edges.append((kx, ky, fx, fy, shin_color, 2.0))

            # Foot
            foot_dir = math.cos(self.facing_angle)
            foot_len = 8
            foot_x_start = fx - foot_len * 0.3
            foot_x_end = fx + foot_len * 0.7 * (1 if foot_dir >= 0 else -1)
            edges.append((foot_x_start, fy, foot_x_end, fy, shin_color, 2.0))

        return edges


# =============================================================================
# CREATURE 3: CentipedeCreature — Chain Spine with Legs
# =============================================================================


class CentipedeCreature(WanderingBody):
    def __init__(self, x, y):
        super().__init__(x, y, speed=1.0)
        self.num_segments = 18
        self.seg_distance = 16
        self.segments = [
            [x - i * self.seg_distance, y] for i in range(self.num_segments)
        ]
        self.hue_offset = 0.0
        self.anim_time = 0.0
        self.base_hue = 0.35  # green
        self.leg_length_1 = 10
        self.leg_length_2 = 10
        self.antenna_length = 20

    def update_animation(self):
        # Head follows WanderingBody position
        self.segments[0][0] = self.x
        self.segments[0][1] = self.y

        # Chain physics
        for i in range(1, self.num_segments):
            dx = self.segments[i - 1][0] - self.segments[i][0]
            dy = self.segments[i - 1][1] - self.segments[i][1]
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            if dist != self.seg_distance:
                ratio = self.seg_distance / dist
                self.segments[i][0] = self.segments[i - 1][0] - dx * ratio
                self.segments[i][1] = self.segments[i - 1][1] - dy * ratio

        # Secondary sinusoidal body undulation
        for i in range(1, self.num_segments):
            dx = self.segments[i][0] - self.segments[i - 1][0]
            dy = self.segments[i][1] - self.segments[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            perp_x = -dy / dist
            perp_y = dx / dist
            offset = math.sin(self.anim_time * 2.0 + i * 0.3) * 3.0
            self.segments[i][0] += perp_x * offset
            self.segments[i][1] += perp_y * offset

        self.anim_time += 0.04
        self.hue_offset += 0.003

    def get_edges(self):
        edges = []
        n = self.num_segments

        # --- Antennae at head ---
        if n >= 2:
            head_x, head_y = self.segments[0]
            s1_x, s1_y = self.segments[1]
            fwd_dx = head_x - s1_x
            fwd_dy = head_y - s1_y
            fwd_dist = math.sqrt(fwd_dx * fwd_dx + fwd_dy * fwd_dy) + 0.001
            fwd_nx = fwd_dx / fwd_dist
            fwd_ny = fwd_dy / fwd_dist

            sway = math.sin(self.anim_time * 3.5) * 0.4

            ant_hue = (self.base_hue + self.hue_offset) % 1.0
            ant_color = hsv_to_hex(ant_hue, 0.7, 0.95)

            # Left antenna
            left_angle = math.atan2(fwd_ny, fwd_nx) + 0.45 + sway
            la_ex = head_x + self.antenna_length * math.cos(left_angle)
            la_ey = head_y + self.antenna_length * math.sin(left_angle)
            edges.append((head_x, head_y, la_ex, la_ey, ant_color, 1.5))

            # Right antenna
            right_angle = math.atan2(fwd_ny, fwd_nx) - 0.45 + sway
            ra_ex = head_x + self.antenna_length * math.cos(right_angle)
            ra_ey = head_y + self.antenna_length * math.sin(right_angle)
            edges.append((head_x, head_y, ra_ex, ra_ey, ant_color, 1.5))

            # Antenna tips — small fork at end
            fork_len = 6
            for base_angle, tip_x, tip_y in [
                (left_angle, la_ex, la_ey),
                (right_angle, ra_ex, ra_ey),
            ]:
                for fork_offset in (-0.3, 0.3):
                    fa = base_angle + fork_offset
                    fx = tip_x + fork_len * math.cos(fa)
                    fy = tip_y + fork_len * math.sin(fa)
                    edges.append((tip_x, tip_y, fx, fy, ant_color, 1.0))

        # --- Spine segments ---
        for i in range(1, n):
            x1, y1 = self.segments[i - 1]
            x2, y2 = self.segments[i]

            t = i / (n - 1)
            hue = (self.base_hue + t * 0.2 + self.hue_offset) % 1.0
            sat = 0.85
            val = 0.85 - t * 0.2

            color = hsv_to_hex(hue, sat, val)
            lw = max(1.0, 3.0 - t * 2.0)

            edges.append((x1, y1, x2, y2, color, lw))

        # --- Legs (segments 2 through n-3) ---
        for i in range(2, n - 2):
            seg_x, seg_y = self.segments[i]

            prev_x, prev_y = self.segments[i - 1]
            spine_dx = seg_x - prev_x
            spine_dy = seg_y - prev_y
            spine_angle = math.atan2(spine_dy, spine_dx)
            perp_angle = spine_angle + math.pi / 2.0

            wave = math.sin(self.anim_time * 3.0 + i * 0.4)

            t = i / (n - 1)
            leg_hue = (self.base_hue + t * 0.2 + self.hue_offset + 0.04) % 1.0
            leg_color = hsv_to_hex(leg_hue, 0.92, 0.75 - t * 0.1)

            swing = wave * 0.4

            for side, sign in [("left", 1), ("right", -1)]:
                leg_angle = perp_angle + sign * swing
                hip_x = seg_x + self.leg_length_1 * math.cos(leg_angle)
                hip_y = seg_y + self.leg_length_1 * math.sin(leg_angle)

                foot_angle = leg_angle + sign * 0.3
                foot_x = hip_x + self.leg_length_2 * math.cos(foot_angle)
                foot_y = hip_y + self.leg_length_2 * math.sin(foot_angle)

                edges.append((seg_x, seg_y, hip_x, hip_y, leg_color, 1.5))
                edges.append((hip_x, hip_y, foot_x, foot_y, leg_color, 1.0))

                # Tiny claw
                claw_len = 4
                for claw_off in (-0.25, 0.25):
                    ca = foot_angle + claw_off
                    cx = foot_x + claw_len * math.cos(ca)
                    cy = foot_y + claw_len * math.sin(ca)
                    edges.append((foot_x, foot_y, cx, cy, leg_color, 0.8))

        return edges


# =============================================================================
# CREATURES VISUALIZATION (wraps all 3 creatures as one viz)
# =============================================================================


class CreaturesViz:
    """Three walking creatures that follow the cursor."""

    name = "Walking Creatures"
    num_lines = 800

    def __init__(self):
        self.cursor = CursorTracker()
        self.creatures = [
            CentipedeCreature(450, 450),
            SpiderCreature(300, 400),
            StickPerson(600, 350),
        ]

    def on_motion(self, event):
        self.cursor.on_motion(event)

    def resize(self, w, h):
        for c in self.creatures:
            c.resize(w, h)

    def update(self):
        self.cursor.update()
        for creature in self.creatures:
            creature.update_position(self.cursor)
            if hasattr(creature, "update_animation"):
                creature.update_animation()

        # Inter-creature repulsion — prevent overlap
        repel_threshold = 150
        repel_strength = 0.04
        n = len(self.creatures)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.creatures[i], self.creatures[j]
                dx = a.x - b.x
                dy = a.y - b.y
                dist = math.sqrt(dx * dx + dy * dy) + 0.001
                if dist < repel_threshold:
                    force = (repel_threshold - dist) * repel_strength
                    nx, ny = dx / dist, dy / dist
                    a.x += nx * force
                    a.y += ny * force
                    b.x -= nx * force
                    b.y -= ny * force

    def get_edges(self, width, height):
        edges = []
        for creature in self.creatures:
            edges.extend(creature.get_edges())
        return edges


# =============================================================================
# NEW VISUALIZATIONS (Version 07)
# =============================================================================


class DNAHelix:
    name = "DNA Double Helix"
    num_lines = 120

    def __init__(self):
        self.angle = 0.0
        self.hue_offset = 0.0
        self.num_points = 30  # per strand
        self.num_rungs = 15

    def update(self):
        self.angle += CONFIG["rotation_speed"] * 1.2
        self.hue_offset += CONFIG["color_cycle_speed"]

    def get_edges(self, width, height):
        edges = []
        cx, cy = width / 2, height / 2
        radius = min(width, height) * 0.08
        total_h = height * 0.7
        half_h = total_h / 2

        n = self.num_points
        turns = 3.0

        strand_points = []
        for strand_idx in range(2):
            phase = strand_idx * math.pi
            pts = []
            for i in range(n):
                t = i / (n - 1)
                angle_helix = t * 2 * math.pi * turns + phase
                x3 = radius * math.cos(angle_helix)
                y3 = t * total_h - half_h
                z3 = radius * math.sin(angle_helix)
                pts.append((x3, y3, z3))
            strand_points.append(pts)

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def rotate_y(x3, y3, z3):
            rx = x3 * cos_a + z3 * sin_a
            ry = y3
            rz = -x3 * sin_a + z3 * cos_a
            return rx, ry, rz

        def project(rx, ry, rz):
            sx = cx + rx
            sy = cy - ry
            return sx, sy

        for strand_idx in range(2):
            pts = strand_points[strand_idx]
            for i in range(n - 1):
                x3a, y3a, z3a = pts[i]
                x3b, y3b, z3b = pts[i + 1]

                rxa, rya, rza = rotate_y(x3a, y3a, z3a)
                rxb, ryb, rzb = rotate_y(x3b, y3b, z3b)

                sxa, sya = project(rxa, rya, rza)
                sxb, syb = project(rxb, ryb, rzb)

                t = i / (n - 1)
                hue = (self.hue_offset + t) % 1.0

                avg_z = (rza + rzb) / 2
                depth_norm = (avg_z + radius) / (2 * radius)
                value = 0.55 + 0.45 * depth_norm
                saturation = 0.85

                color = hsv_to_hex(hue, saturation, value)
                lw = 1.5 + depth_norm * 1.0

                edges.append((sxa, sya, sxb, syb, color, lw))

        rung_indices = [
            int(i * (n - 1) / (self.num_rungs - 1)) for i in range(self.num_rungs)
        ]

        for rung_i, pt_idx in enumerate(rung_indices):
            pt_idx = min(pt_idx, n - 1)

            x3a, y3a, z3a = strand_points[0][pt_idx]
            x3b, y3b, z3b = strand_points[1][pt_idx]

            rxa, rya, rza = rotate_y(x3a, y3a, z3a)
            rxb, ryb, rzb = rotate_y(x3b, y3b, z3b)

            sxa, sya = project(rxa, rya, rza)
            sxb, syb = project(rxb, ryb, rzb)

            msx = (sxa + sxb) / 2
            msy = (sya + syb) / 2

            t = pt_idx / max(n - 1, 1)
            hue = (self.hue_offset + t + 0.15) % 1.0

            avg_z = (rza + rzb) / 2
            depth_norm = (avg_z + radius) / (2 * radius)
            value = 0.50 + 0.40 * depth_norm
            saturation = 0.55

            color = hsv_to_hex(hue, saturation, value)
            lw = 1.0 + depth_norm * 0.8

            edges.append((sxa, sya, msx, msy, color, lw))
            edges.append((msx, msy, sxb, syb, color, lw))

        return edges


class LorenzAttractor:
    name = "Lorenz Attractor"
    num_lines = 400

    def __init__(self):
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.x = 1.0
        self.y = 1.0
        self.z = 1.0
        self.trail = []
        self.max_trail = 400
        self.angle = 0.0
        self.hue_offset = 0.0

    def update(self):
        dt = 0.005
        for _ in range(4):
            dx = self.sigma * (self.y - self.x)
            dy = self.x * (self.rho - self.z) - self.y
            dz = self.x * self.y - self.beta * self.z
            self.x += dx * dt
            self.y += dy * dt
            self.z += dz * dt
        self.trail.append((self.x, self.y, self.z))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        self.angle += CONFIG["rotation_speed"] * 0.5
        self.hue_offset += CONFIG["color_cycle_speed"]

    def get_edges(self, width, height):
        edges = []
        if len(self.trail) < 2:
            return edges

        cx, cy = width / 2, height / 2
        scale = min(width, height) * 0.008

        center_x = 0.0
        center_y = 0.0
        center_z = 25.0

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def rotate_y(lx, ly, lz):
            rx = lx * cos_a + lz * sin_a
            ry = ly
            rz = -lx * sin_a + lz * cos_a
            return rx, ry, rz

        def project(rx, ry, rz):
            sx = cx + rx * scale
            sy = cy - ry * scale
            return sx, sy

        n = len(self.trail)
        for i in range(n - 1):
            lx0, ly0, lz0 = self.trail[i]
            lx1, ly1, lz1 = self.trail[i + 1]

            lx0 -= center_x
            ly0 -= center_y
            lz0 -= center_z
            lx1 -= center_x
            ly1 -= center_y
            lz1 -= center_z

            rx0, ry0, rz0 = rotate_y(lx0, ly0, lz0)
            rx1, ry1, rz1 = rotate_y(lx1, ly1, lz1)

            sx0, sy0 = project(rx0, ry0, rz0)
            sx1, sy1 = project(rx1, ry1, rz1)

            t = i / max(n - 2, 1)
            hue = (self.hue_offset + t * 0.75) % 1.0

            value = 0.30 + 0.70 * t
            saturation = 0.80 + 0.20 * t

            avg_z = (rz0 + rz1) / 2
            depth_norm = (avg_z + 25.0) / 50.0
            depth_norm = max(0.0, min(1.0, depth_norm))
            value = value * (0.7 + 0.3 * depth_norm)
            value = max(0.0, min(1.0, value))

            color = hsv_to_hex(hue, saturation, value)

            lw = 1.0 + t * 1.5

            edges.append((sx0, sy0, sx1, sy1, color, lw))

        return edges


class PendulumWave:
    name = "Pendulum Wave"
    num_lines = 80

    def __init__(self):
        self.time = 0.0
        self.hue_offset = 0.0
        self.n = 15
        self.amplitude = 0.8
        self.base_period = 2.5

    def update(self):
        self.time += 1.0 / 30.0
        self.hue_offset += CONFIG["color_cycle_speed"]

    def get_edges(self, width, height):
        edges = []
        bobs = []

        pivot_y = height * 0.15
        cross_size = max(4, height * 0.012)

        for i in range(self.n):
            pivot_x = width * 0.05 + (width * 0.90) * i / (self.n - 1)

            length = height * (0.35 + 0.20 * i / (self.n - 1))

            period = self.base_period * (1.0 + i / 30.0)

            angle = self.amplitude * math.sin(2.0 * math.pi * self.time / period)

            bob_x = pivot_x + length * math.sin(angle)
            bob_y = pivot_y + length * math.cos(angle)
            bobs.append((bob_x, bob_y))

            hue = (i / self.n + self.hue_offset) % 1.0
            color = hsv_to_hex(hue, 0.85, 0.95)

            edges.append((pivot_x, pivot_y, bob_x, bob_y, color, 2))

            edges.append(
                (bob_x - cross_size, bob_y, bob_x + cross_size, bob_y, color, 2)
            )
            edges.append(
                (bob_x, bob_y - cross_size, bob_x, bob_y + cross_size, color, 2)
            )

        for i in range(len(bobs) - 1):
            x1, y1 = bobs[i]
            x2, y2 = bobs[i + 1]
            wave_hue = (self.hue_offset + 0.5) % 1.0
            wave_color = hsv_to_hex(wave_hue, 0.4, 1.0)
            edges.append((x1, y1, x2, y2, wave_color, 1))

        return edges


class AuroraBorealis:
    name = "Aurora Borealis"
    num_lines = 250

    def __init__(self):
        self.time = 0.0
        self.hue_offset = 0.0
        self.num_curtains = 6
        self.curtain_offsets = [
            random.uniform(0, 2 * math.pi) for _ in range(self.num_curtains)
        ]
        self.curtain_x = [random.uniform(0.05, 0.95) for _ in range(self.num_curtains)]
        self.curtain_speed = [
            random.uniform(0.6, 1.4) for _ in range(self.num_curtains)
        ]
        self.curtain_width = [
            random.uniform(0.06, 0.14) for _ in range(self.num_curtains)
        ]
        self.curtain_depth = [
            random.uniform(0.45, 0.72) for _ in range(self.num_curtains)
        ]
        self.curtain_wave_offset = [
            random.uniform(0, 2 * math.pi) for _ in range(self.num_curtains)
        ]
        self.curtain_hues = [
            random.uniform(0.25, 0.85) for _ in range(self.num_curtains)
        ]

    def update(self):
        self.time += 0.02
        self.hue_offset += CONFIG["color_cycle_speed"] * 0.3

    def get_edges(self, width, height):
        edges = []
        segments_per_curtain = 25

        for c in range(self.num_curtains):
            offset = self.curtain_offsets[c]
            speed = self.curtain_speed[c]
            t = self.time * speed

            center_x = (
                self.curtain_x[c] * width + math.sin(t * 0.4 + offset) * width * 0.03
            )

            half_w = self.curtain_width[c] * width

            top_y_base = height * 0.04

            bottom_depth = self.curtain_depth[c]
            wave_off = self.curtain_wave_offset[c]

            for s in range(segments_per_curtain):
                t_s = s / (segments_per_curtain - 1)

                seg_x = center_x - half_w + t_s * 2.0 * half_w
                warp = math.sin(t * 0.7 + t_s * math.pi * 2.0 + offset) * half_w * 0.25
                seg_x += warp

                top_y = (
                    top_y_base + math.sin(t * 0.9 + t_s * 3.5 + offset) * height * 0.03
                )

                bottom_phase = t * 1.1 + t_s * math.pi * 4.0 + wave_off
                bottom_y = (
                    height * bottom_depth
                    + math.sin(bottom_phase) * height * 0.08
                    + math.cos(bottom_phase * 0.6 + 1.2) * height * 0.04
                )

                top_y = max(0, min(height * 0.12, top_y))
                bottom_y = min(height * 0.90, bottom_y)

                if bottom_y <= top_y:
                    continue

                t_mid = t_s * 0.4
                hue = (self.curtain_hues[c] + t_mid * 0.15 + self.hue_offset) % 1.0

                center_dist = abs(t_s - 0.5) * 2.0
                shimmer = 0.85 + 0.15 * math.sin(t * 1.3 + t_s * 5.0 + offset * 0.7)
                brightness = (1.0 - center_dist * 0.45) * shimmer
                brightness = max(0.25, min(1.0, brightness))

                saturation = 0.85
                color = hsv_to_hex(hue, saturation, brightness)

                lw = 2 if center_dist < 0.5 else 1

                edges.append((seg_x, top_y, seg_x, bottom_y, color, lw))

                if s % 2 == 0:
                    glow_hue = (self.curtain_hues[c] + self.hue_offset) % 1.0
                    glow_color = hsv_to_hex(glow_hue, 0.70, 0.45)
                    glow_bottom = bottom_y
                    glow_top = bottom_y - (bottom_y - top_y) * 0.25
                    edges.append(
                        (seg_x + 2.0, glow_top, seg_x + 2.0, glow_bottom, glow_color, 1)
                    )

        return edges


class WireframeTerrain:
    name = "Wireframe Terrain"
    num_lines = 600

    def __init__(self):
        self.time = 0.0
        self.hue_offset = 0.0
        self.cols = 20
        self.rows = 15
        self.scroll_z = 0.0

    def update(self):
        self.time += 0.03
        self.scroll_z += 0.05
        self.hue_offset += CONFIG["color_cycle_speed"]

    def _get_height(self, gx, gz):
        col_center = (self.cols - 1) / 2.0
        dist_from_center = abs(gx - col_center) / col_center
        amp = 1.8 * (1.0 - 0.6 * dist_from_center)

        x_freq = 0.45
        z_freq = 0.35
        h = (
            amp
            * math.sin(x_freq * gx + self.time)
            * math.cos(z_freq * gz + self.time * 0.7)
        )
        h += (
            0.6
            * math.sin(0.7 * gx - self.time * 0.5)
            * math.sin(0.5 * gz + self.time * 0.4)
        )
        return h

    def _project(self, wx, wy, wz, cx, cy, fov_scale):
        if wz < 0.1:
            wz = 0.1
        inv_z = fov_scale / wz
        sx = cx + wx * inv_z
        sy = cy + wy * inv_z
        return sx, sy

    def get_edges(self, width, height):
        edges = []

        cx = width / 2.0
        cy = height * 0.35

        z_near = 2.0
        z_far = 14.0
        x_span = 10.0

        fov_scale = height * 0.55

        z_range = z_far - z_near
        row_step = z_range / (self.rows - 1)

        pts = []
        for row in range(self.rows):
            z_offset = self.scroll_z % row_step
            wz = z_far - row * row_step + z_offset
            if wz < z_near:
                wz = z_near
            if wz > z_far + row_step:
                wz = z_far + row_step

            t_depth = (wz - z_near) / z_range
            t_depth = max(0.0, min(1.0, t_depth))

            row_pts = []
            for col in range(self.cols):
                gx = col
                gz = wz / row_step + self.scroll_z * 0.12

                wx = (col / (self.cols - 1) - 0.5) * x_span
                raw_h = self._get_height(gx, gz)
                wy = raw_h * (0.5 + 0.5 * t_depth)

                sx, sy = self._project(wx, wy, wz, cx, cy, fov_scale)
                row_pts.append((sx, sy, raw_h, t_depth))
            pts.append(row_pts)

        for row in range(self.rows):
            for col in range(self.cols - 1):
                sx1, sy1, h1, td1 = pts[row][col]
                sx2, sy2, h2, td2 = pts[row][col + 1]

                avg_h = (h1 + h2) / 2.0
                avg_td = (td1 + td2) / 2.0

                h_norm = (avg_h + 2.4) / 4.8
                h_norm = max(0.0, min(1.0, h_norm))

                hue = self.hue_offset + 0.65 - h_norm * 0.35
                hue = hue % 1.0
                sat = 0.8 + 0.2 * h_norm
                val = 0.95 - 0.55 * avg_td
                val = max(0.2, val)

                color = hsv_to_hex(hue, sat, val)
                lw = max(0.5, 2.0 - 1.5 * avg_td)
                edges.append((sx1, sy1, sx2, sy2, color, lw))

        for col in range(self.cols):
            for row in range(self.rows - 1):
                sx1, sy1, h1, td1 = pts[row][col]
                sx2, sy2, h2, td2 = pts[row + 1][col]

                avg_h = (h1 + h2) / 2.0
                avg_td = (td1 + td2) / 2.0

                h_norm = (avg_h + 2.4) / 4.8
                h_norm = max(0.0, min(1.0, h_norm))

                hue = self.hue_offset + 0.65 - h_norm * 0.35
                hue = hue % 1.0
                sat = 0.75
                val = 0.9 - 0.55 * avg_td
                val = max(0.15, val)

                color = hsv_to_hex(hue, sat, val)
                lw = max(0.4, 1.5 - 1.1 * avg_td)
                edges.append((sx1, sy1, sx2, sy2, color, lw))

        return edges


class Lissajous3D:
    name = "Lissajous 3D"
    num_lines = 300

    _FREQ_A = [2.0, 3.0, 4.0, 5.0]
    _FREQ_B = [1.0, 3.0, 5.0]
    _FREQ_C = [2.0, 4.0, 6.0]

    def __init__(self):
        self.angle_y = 0.0
        self.angle_x = 0.0
        self.hue_offset = 0.0
        self.morph_time = 0.0
        self.morph_duration = 5.0

        self.freq_a = 3.0
        self.freq_b = 2.0
        self.freq_c = 4.0

        self.target_a = 5.0
        self.target_b = 3.0
        self.target_c = 2.0

        self.num_points = 300

        self.delta = 0.0
        self.phi = 0.0

    def _pick_new_targets(self):
        candidates_a = [f for f in self._FREQ_A if abs(f - self.freq_a) > 0.5]
        candidates_b = [f for f in self._FREQ_B if abs(f - self.freq_b) > 0.5]
        candidates_c = [f for f in self._FREQ_C if abs(f - self.freq_c) > 0.5]

        self.target_a = (
            random.choice(candidates_a) if candidates_a else random.choice(self._FREQ_A)
        )
        self.target_b = (
            random.choice(candidates_b) if candidates_b else random.choice(self._FREQ_B)
        )
        self.target_c = (
            random.choice(candidates_c) if candidates_c else random.choice(self._FREQ_C)
        )

        self.delta = random.uniform(0.0, math.pi)
        self.phi = random.uniform(0.0, math.pi)

    def update(self):
        self.angle_y += CONFIG["rotation_speed"] * 1.5
        self.angle_x += CONFIG["rotation_speed"] * 0.7
        self.hue_offset += CONFIG["color_cycle_speed"]

        dt = 1.0 / 30.0
        self.morph_time += dt

        if self.morph_time >= self.morph_duration:
            self.freq_a = self.target_a
            self.freq_b = self.target_b
            self.freq_c = self.target_c
            self.morph_time = 0.0
            self._pick_new_targets()
        else:
            self.freq_a = (
                self.freq_a
                + (self.target_a - self.freq_a) * (dt / self.morph_duration) * 3.0
            )
            self.freq_b = (
                self.freq_b
                + (self.target_b - self.freq_b) * (dt / self.morph_duration) * 3.0
            )
            self.freq_c = (
                self.freq_c
                + (self.target_c - self.freq_c) * (dt / self.morph_duration) * 3.0
            )

    def _rotate_y(self, x, y, z, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = cos_a * x + sin_a * z
        ry = y
        rz = -sin_a * x + cos_a * z
        return rx, ry, rz

    def _rotate_x(self, x, y, z, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = x
        ry = cos_a * y - sin_a * z
        rz = sin_a * y + cos_a * z
        return rx, ry, rz

    def _project(self, x, y, z, cx, cy, scale):
        distance = 4.0
        denom = distance - z
        if denom < 0.01:
            denom = 0.01
        inv = distance / denom
        sx = cx + x * scale * inv
        sy = cy - y * scale * inv
        return sx, sy

    def get_edges(self, width, height):
        edges = []

        cx = width / 2.0
        cy = height / 2.0
        scale = min(width, height) * 0.3

        n = self.num_points
        two_pi = 2.0 * math.pi

        points_3d = []
        for i in range(n + 1):
            t = two_pi * i / n
            x = math.sin(self.freq_a * t + self.delta)
            y = math.sin(self.freq_b * t)
            z = math.sin(self.freq_c * t + self.phi)

            x, y, z = self._rotate_y(x, y, z, self.angle_y)
            x, y, z = self._rotate_x(x, y, z, self.angle_x)

            points_3d.append((x, y, z))

        screen_pts = []
        for x, y, z in points_3d:
            sx, sy = self._project(x, y, z, cx, cy, scale)
            screen_pts.append((sx, sy))

        for i in range(n):
            sx1, sy1 = screen_pts[i]
            sx2, sy2 = screen_pts[i + 1]

            hue = (self.hue_offset + i / n) % 1.0
            color = hsv_to_hex(hue, 0.9, 0.95)
            edges.append((sx1, sy1, sx2, sy2, color, 2))

        return edges


class MagneticField:
    name = "Magnetic Field"
    num_lines = 500

    def __init__(self):
        self.angle = 0.0
        self.hue_offset = 0.0
        self.num_field_lines = 14
        self.steps_per_line = 60

    def update(self):
        self.angle += CONFIG["rotation_speed"] * 0.6
        self.hue_offset += CONFIG["color_cycle_speed"]

    def _field_at(self, px, py, p1x, p1y, p2x, p2y):
        dx1 = px - p1x
        dy1 = py - p1y
        r1sq = dx1 * dx1 + dy1 * dy1
        if r1sq < 1.0:
            r1sq = 1.0
        r1 = math.sqrt(r1sq)
        fx = dx1 / (r1sq * r1)
        fy = dy1 / (r1sq * r1)

        dx2 = px - p2x
        dy2 = py - p2y
        r2sq = dx2 * dx2 + dy2 * dy2
        if r2sq < 1.0:
            r2sq = 1.0
        r2 = math.sqrt(r2sq)
        fx -= dx2 / (r2sq * r2)
        fy -= dy2 / (r2sq * r2)

        return fx, fy

    def _trace_field_line(
        self, sx, sy, p1x, p1y, p2x, p2y, width, height, reverse=False
    ):
        points = [(sx, sy)]
        step = 8.0
        x, y = sx, sy
        margin = 20.0
        for _ in range(self.steps_per_line):
            fx, fy = self._field_at(x, y, p1x, p1y, p2x, p2y)
            if reverse:
                fx, fy = -fx, -fy
            mag = math.sqrt(fx * fx + fy * fy)
            if mag < 1e-10:
                break
            nx = fx / mag
            ny = fy / mag
            x += nx * step
            y += ny * step
            points.append((x, y))
            if x < -margin or x > width + margin or y < -margin or y > height + margin:
                break
            dx2 = x - p2x
            dy2 = y - p2y
            if math.sqrt(dx2 * dx2 + dy2 * dy2) < step * 2.5:
                break
        return points

    def _color_for_segment(self, seg_index, total_segs):
        t = seg_index / max(total_segs - 1, 1)
        mid_t = 1.0 - abs(t - 0.5) * 2.0
        h = (0.0 + mid_t * 0.15 + self.hue_offset) % 1.0
        s = 0.9
        v = 0.7 + mid_t * 0.25
        return hsv_to_hex(h, s, min(v, 1.0))

    def get_edges(self, width, height):
        edges = []
        cx, cy = width / 2, height / 2
        orbit_r = min(width, height) * 0.18

        p1x = cx + orbit_r * math.cos(self.angle)
        p1y = cy + orbit_r * math.sin(self.angle)
        p2x = cx - orbit_r * math.cos(self.angle)
        p2y = cy - orbit_r * math.sin(self.angle)

        for i in range(self.num_field_lines):
            start_angle = 2.0 * math.pi * i / self.num_field_lines
            start_r = 18.0
            sx = p1x + start_r * math.cos(start_angle)
            sy = p1y + start_r * math.sin(start_angle)

            points = self._trace_field_line(
                sx, sy, p1x, p1y, p2x, p2y, width, height, reverse=False
            )

            for j in range(len(points) - 1):
                x1, y1 = points[j]
                x2, y2 = points[j + 1]
                color = self._color_for_segment(j, len(points))
                edges.append((x1, y1, x2, y2, color, 2))

        starburst_rays = 6
        ray_len_outer = 14.0
        ray_len_inner = 4.0
        for pole_x, pole_y, is_north in [(p1x, p1y, True), (p2x, p2y, False)]:
            if is_north:
                h = (0.0 + self.hue_offset) % 1.0
            else:
                h = (0.6 + self.hue_offset) % 1.0
            pole_color = hsv_to_hex(h, 1.0, 1.0)
            for k in range(starburst_rays):
                ray_angle = math.pi * k / starburst_rays
                dx = math.cos(ray_angle)
                dy = math.sin(ray_angle)
                x1 = pole_x + dx * ray_len_inner
                y1 = pole_y + dy * ray_len_inner
                x2 = pole_x + dx * ray_len_outer
                y2 = pole_y + dy * ray_len_outer
                edges.append((x1, y1, x2, y2, pole_color, 3))
                x1b = pole_x - dx * ray_len_inner
                y1b = pole_y - dy * ray_len_inner
                x2b = pole_x - dx * ray_len_outer
                y2b = pole_y - dy * ray_len_outer
                edges.append((x1b, y1b, x2b, y2b, pole_color, 3))

        return edges


class FractalTree:
    name = "Fractal Tree"
    num_lines = 1500

    def __init__(self):
        self.time = 0.0
        self.hue_offset = 0.0
        self.cycle_duration = 12.0
        self.max_depth = 8
        self.sway_speed = 0.4
        self.sway_amplitude = 0.04

        self.tree_nodes = []
        self._precompute_tree()

    def update(self):
        self.time += 1.0 / 30.0
        self.hue_offset += CONFIG["color_cycle_speed"]

    def _precompute_tree(self):
        random.seed(42)

        trunk_node = {
            "depth": 0,
            "parent_idx": -1,
            "angle_from_parent": 0.0,
            "length_ratio": 1.0,
            "width_at_depth": 3.5,
            "child_indices": [],
        }
        self.tree_nodes = [trunk_node]

        queue = [0]
        while queue:
            idx = queue.pop(0)
            node = self.tree_nodes[idx]
            if node["depth"] >= self.max_depth - 1:
                continue

            if node["depth"] < 3:
                n_children = random.choice([2, 3])
            else:
                n_children = 2

            spread_base = 0.45 + random.uniform(-0.05, 0.05)
            child_length_mult = 0.68 + random.uniform(-0.04, 0.04)

            if n_children == 2:
                offsets = [-spread_base, spread_base]
            else:
                offsets = [-spread_base, 0.0, spread_base]

            for offset in offsets:
                angle_var = random.uniform(-0.08, 0.08)
                length_var = random.uniform(-0.04, 0.04)
                child_width = max(0.5, node["width_at_depth"] - 0.38)
                child_node = {
                    "depth": node["depth"] + 1,
                    "parent_idx": idx,
                    "angle_from_parent": offset + angle_var,
                    "length_ratio": (
                        node["length_ratio"] * (child_length_mult + length_var)
                    ),
                    "width_at_depth": child_width,
                    "child_indices": [],
                }
                child_idx = len(self.tree_nodes)
                self.tree_nodes.append(child_node)
                node["child_indices"].append(child_idx)
                queue.append(child_idx)

        random.seed()

    def _growth_factor(self, node_depth, cycle_t):
        max_d = self.max_depth

        if cycle_t < 6.0:
            appear_t = node_depth * (6.0 / max_d)
            if cycle_t < appear_t:
                return 0.0
            grow_duration = 0.8
            return min(1.0, (cycle_t - appear_t) / grow_duration)

        elif cycle_t < 9.0:
            return 1.0

        else:
            shed_start = 9.0 + (max_d - 1 - node_depth) * (3.0 / max_d)
            shed_duration = 3.0 / max_d
            if cycle_t < shed_start:
                return 1.0
            return max(0.0, 1.0 - (cycle_t - shed_start) / shed_duration)

    def _branch_color(self, depth, cycle_t):
        max_d = self.max_depth
        t_norm = depth / max(max_d - 1, 1)

        if cycle_t < 6.0:
            h = (0.08 + t_norm * 0.22 + self.hue_offset) % 1.0
            s = 0.7 + t_norm * 0.2
            v = 0.5 + t_norm * 0.35
        elif cycle_t < 9.0:
            h = (0.08 + t_norm * 0.28 + self.hue_offset) % 1.0
            s = 0.75
            v = 0.55 + t_norm * 0.35
        else:
            shed_t = (cycle_t - 9.0) / 3.0
            tip_hue_shift = t_norm * shed_t * (-0.17)
            h = (0.08 + t_norm * 0.28 + tip_hue_shift + self.hue_offset) % 1.0
            s = 0.8
            v = 0.5 + t_norm * 0.35

        return hsv_to_hex(h % 1.0, min(s, 1.0), min(v, 1.0))

    def get_edges(self, width, height):
        edges = []
        cycle_t = self.time % self.cycle_duration

        base_x = width / 2
        base_y = height * 0.92
        trunk_length = height * 0.22

        sway_t = self.time * self.sway_speed

        stack = [(0, base_x, base_y, -math.pi / 2)]

        while stack:
            node_idx, sx, sy, parent_angle = stack.pop()
            node = self.tree_nodes[node_idx]
            depth = node["depth"]

            if depth == 0:
                branch_angle = -math.pi / 2
            else:
                branch_angle = parent_angle + node["angle_from_parent"]

            sway_phase = depth * 0.35 + node_idx * 0.07
            sway = self.sway_amplitude * depth * math.sin(sway_t + sway_phase)
            branch_angle += sway

            gf = self._growth_factor(depth, cycle_t)
            if gf <= 0.0:
                continue

            branch_len = trunk_length * node["length_ratio"] * gf

            ex = sx + math.cos(branch_angle) * branch_len
            ey = sy + math.sin(branch_angle) * branch_len

            color = self._branch_color(depth, cycle_t)
            lw = node["width_at_depth"]

            edges.append((sx, sy, ex, ey, color, lw))

            for child_idx in node["child_indices"]:
                stack.append((child_idx, ex, ey, branch_angle))

            if (
                depth >= self.max_depth - 2
                and gf > 0.7
                and cycle_t > 3.0
                and cycle_t < 10.5
            ):
                leaf_count = 3
                leaf_len = branch_len * 0.35
                for lk in range(leaf_count):
                    leaf_angle = branch_angle + (lk - leaf_count // 2) * 0.55
                    lx1 = ex
                    ly1 = ey
                    lx2 = ex + math.cos(leaf_angle) * leaf_len
                    ly2 = ey + math.sin(leaf_angle) * leaf_len
                    if cycle_t < 6.0:
                        leaf_h = (0.28 + self.hue_offset) % 1.0
                    elif cycle_t < 9.0:
                        leaf_h = (0.25 + self.hue_offset) % 1.0
                    else:
                        shed_frac = (cycle_t - 9.0) / 3.0
                        leaf_h = (0.25 - shed_frac * 0.17 + self.hue_offset) % 1.0
                    leaf_color = hsv_to_hex(leaf_h % 1.0, 0.9, 0.95)
                    edges.append((lx1, ly1, lx2, ly2, leaf_color, 1))

        return edges


class MathJellyfish:
    """Bioluminescent mathematical jellyfish made of parametric curves."""

    name = "Math Jellyfish"
    num_lines = 600

    # Tentacle configuration — precomputed at init for performance
    _NUM_TENTACLES = 12
    _TENTACLE_SEGMENTS = 28
    _NUM_ORAL = 4
    _ORAL_SEGMENTS = 15
    _BELL_ARCS = 8
    _ARC_SEGMENTS = 20

    def __init__(self):
        self.time = 0.0
        self.hue_offset = 0.0

        # Jellyfish position — starts roughly centered
        self.cx = 0.0  # will be set relative to width in get_edges
        self.cy = 0.0  # will be set relative to height in get_edges
        self._cx_init = False  # flag to initialize position on first call

        # Per-tentacle parameters (randomised once, fixed across frames)
        rng = random.Random(42)  # deterministic seed for reproducibility
        self._tentacle_freq = [
            rng.uniform(1.5, 3.0) for _ in range(self._NUM_TENTACLES)
        ]
        self._tentacle_wave_speed = [
            rng.uniform(2.5, 4.5) for _ in range(self._NUM_TENTACLES)
        ]
        self._tentacle_base_amp = [
            rng.uniform(0.5, 1.2) for _ in range(self._NUM_TENTACLES)
        ]

        # Per-oral-arm parameters
        self._oral_freq = [rng.uniform(1.0, 1.8) for _ in range(self._NUM_ORAL)]
        self._oral_wave_speed = [rng.uniform(1.2, 2.0) for _ in range(self._NUM_ORAL)]

    def update(self):
        self.time += 1.0 / 30.0  # 30 fps → seconds
        self.hue_offset += CONFIG["color_cycle_speed"]

    def _bell_radius(self, base_r):
        """Pulsing bell radius — contracts and expands rhythmically."""
        pulse = math.sin(self.time * 1.5) * 0.12  # ±12 % oscillation
        return base_r * (1.0 + pulse)

    def _tentacle_points(
        self,
        anchor_x,
        anchor_y,
        tentacle_idx,
        seg_count,
        seg_spacing,
        base_amplitude,
        freq,
        wave_speed,
        bell_pulse_factor,
    ):
        """Return list of (x, y) points for one tentacle chain."""
        points = [(anchor_x, anchor_y)]
        x, y = anchor_x, anchor_y
        t = self.time
        phase = (tentacle_idx / max(self._NUM_TENTACLES, 1)) * 2.0 * math.pi

        for s in range(1, seg_count + 1):
            frac = s / seg_count  # 0 → 1 from base to tip
            # Amplitude grows towards tip; bell contraction spreads tentacles
            amp = base_amplitude * (0.4 + frac * 1.6) * (1.0 + bell_pulse_factor * 0.3)
            dx = amp * math.sin(freq * s * 0.35 + phase + t * wave_speed)
            # Secondary wiggle for organic feel
            dx += (amp * 0.25) * math.sin(
                freq * 1.7 * s * 0.35 + phase + t * wave_speed * 1.3 + 1.0
            )
            dy = seg_spacing
            x += dx
            y += dy
            points.append((x, y))
        return points

    def get_edges(self, width, height):
        edges = []
        t = self.time

        # Initialise position on first call
        if not self._cx_init:
            self.cx = width / 2.0
            self.cy = height / 2.0
            self._cx_init = True

        # Pulse-driven propulsion — bell contraction thrusts upward
        pulse = math.sin(t * 1.5)
        thrust = max(0, -pulse)  # positive only when contracting
        self.cy -= 0.15 + thrust * 1.8  # gentle drift + strong pulse thrust
        # Horizontal sway driven by pulse impulse (lateral kick from contractions)
        pulse_kick = math.cos(t * 1.5) * 0.9
        self.cx += pulse_kick * math.sin(t * 0.13)

        # Wrap: when jellyfish drifts fully off the top, reappear at bottom
        base_r = min(width, height) * 0.18
        br = self._bell_radius(base_r)
        if self.cy < -br * 2.0:
            self.cy = height + br

        # Keep horizontal within screen with gentle bounce
        if self.cx < width * 0.1:
            self.cx = width * 0.1
        elif self.cx > width * 0.9:
            self.cx = width * 0.9

        cx, cy = self.cx, self.cy
        bell_cy = cy - br * 0.3  # bell center slightly above drift point

        # Bell pulse factor (positive when contracting)
        bell_pulse_factor = -math.sin(t * 1.5)  # in phase with pulse

        # ------------------------------------------------------------------ #
        # 1. BELL ARCS — semi-circular dome, 8 concentric arcs               #
        # ------------------------------------------------------------------ #
        arc_r_min = br * 0.30
        arc_r_max = br * 1.00

        for arc_i in range(self._BELL_ARCS):
            frac = arc_i / max(self._BELL_ARCS - 1, 1)  # 0 … 1
            r = arc_r_min + frac * (arc_r_max - arc_r_min)

            # Hue: inner arcs more cyan, outer more blue
            hue = (0.55 + frac * 0.10 + self.hue_offset) % 1.0
            sat = 0.55 + frac * 0.20
            val = 0.95 - frac * 0.10  # outer arcs slightly dimmer

            # θ spans upper semi-circle [0.1π … 0.9π]
            theta_start = 0.10 * math.pi
            theta_end = 0.90 * math.pi
            n = self._ARC_SEGMENTS

            prev_x = cx + r * math.cos(theta_start)
            prev_y = bell_cy - r * math.sin(theta_start)

            for seg in range(1, n + 1):
                theta = theta_start + (theta_end - theta_start) * seg / n
                nx = cx + r * math.cos(theta)
                ny = bell_cy - r * math.sin(theta)

                # Undulating rim for living texture — only on the outermost arc
                if arc_i == self._BELL_ARCS - 1:
                    ripple = br * 0.03 * math.sin(8 * theta + t * 3.0)
                    prev_x_r = prev_x + ripple * math.cos(theta - 0.5 * math.pi)
                    prev_y_r = prev_y + ripple * math.sin(theta - 0.5 * math.pi)
                    nx_r = nx + ripple * math.cos(theta - 0.5 * math.pi)
                    ny_r = ny + ripple * math.sin(theta - 0.5 * math.pi)
                    color = hsv_to_hex(hue, sat, val)
                    lw = 1.5 if arc_i < self._BELL_ARCS - 2 else 2.0
                    edges.append((prev_x_r, prev_y_r, nx_r, ny_r, color, lw))
                    prev_x, prev_y = nx, ny
                else:
                    color = hsv_to_hex(hue, sat, val)
                    lw = 1.5 if arc_i < self._BELL_ARCS - 2 else 2.0
                    edges.append((prev_x, prev_y, nx, ny, color, lw))
                    prev_x, prev_y = nx, ny

        # ------------------------------------------------------------------ #
        # 2. OUTER TENTACLES — 12 from the bell's lower rim                  #
        # ------------------------------------------------------------------ #
        seg_spacing = br * 0.14  # vertical step per segment
        num_t = self._NUM_TENTACLES

        for ti in range(num_t):
            # Anchor points evenly spaced along bottom of bell (θ from 0 to π)
            angle = math.pi * (ti + 0.5) / num_t  # [~8°..~172°] spread
            anchor_x = cx + br * math.cos(math.pi - angle)
            anchor_y = bell_cy + br * math.sin(angle) * 0.4  # flatten spread

            freq = self._tentacle_freq[ti]
            wave_speed = self._tentacle_wave_speed[ti]
            base_amp = self._tentacle_base_amp[ti] * br * 0.06

            pts = self._tentacle_points(
                anchor_x,
                anchor_y,
                ti,
                self._TENTACLE_SEGMENTS,
                seg_spacing,
                base_amp,
                freq,
                wave_speed,
                bell_pulse_factor,
            )

            for seg in range(len(pts) - 1):
                frac = seg / max(self._TENTACLE_SEGMENTS - 1, 1)  # 0=base 1=tip
                # Hue shifts from cyan at base to blue-violet at tips
                hue = (0.55 + frac * 0.15 + self.hue_offset) % 1.0
                sat = 0.70 + frac * 0.20
                val = 0.90 - frac * 0.60  # bright base → dim tip
                val = max(0.05, val)
                color = hsv_to_hex(hue, sat, val)
                lw = 2.0 - frac * 1.0  # 2px at base → 1px at tip
                lw = max(1.0, lw)

                x1, y1 = pts[seg]
                x2, y2 = pts[seg + 1]
                edges.append((x1, y1, x2, y2, color, lw))

        # ------------------------------------------------------------------ #
        # 3. ORAL ARMS — 4 thicker, shorter tentacles from bell center       #
        # ------------------------------------------------------------------ #
        num_o = self._NUM_ORAL
        oral_seg_spacing = br * 0.16
        oral_base_amp = br * 0.10

        for oi in range(num_o):
            # Evenly spread around center bottom
            angle = (oi / num_o) * 2.0 * math.pi
            spread = br * 0.12
            anchor_x = cx + spread * math.cos(angle)
            anchor_y = bell_cy + br * 0.35  # just below dome center

            freq = self._oral_freq[oi]
            wave_speed = self._oral_wave_speed[oi]

            pts = self._tentacle_points(
                anchor_x,
                anchor_y,
                oi,
                self._ORAL_SEGMENTS,
                oral_seg_spacing,
                oral_base_amp,
                freq,
                wave_speed,
                bell_pulse_factor,
            )

            for seg in range(len(pts) - 1):
                frac = seg / max(self._ORAL_SEGMENTS - 1, 1)
                hue = (0.50 + frac * 0.10 + self.hue_offset) % 1.0
                sat = 0.65
                val = 0.95 - frac * 0.45
                val = max(0.10, val)
                color = hsv_to_hex(hue, sat, val)
                lw = 2.5 - frac * 1.0
                lw = max(1.5, lw)

                x1, y1 = pts[seg]
                x2, y2 = pts[seg + 1]
                edges.append((x1, y1, x2, y2, color, lw))

        return edges


# =============================================================================
# SPECTACULAR 3D VISUALS (Version 08)
# =============================================================================


class OrbitalLattice:
    """Nested orbital cages with shifting 3D cross-links."""

    name = "Orbital Lattice"
    num_lines = 320

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = 0.42
        self.angle_y = 0.0
        self.angle_z = 0.18
        self.nodes_per_ring = 18
        self.rings = []
        for idx in range(5):
            self.rings.append(
                {
                    "radius": 0.72 + idx * 0.3,
                    "tilt_x": random.uniform(-0.9, 0.9),
                    "tilt_z": random.uniform(-1.0, 1.0),
                    "phase": random.random() * 2 * math.pi,
                    "speed": random.uniform(0.7, 1.35) * (1 if idx % 2 == 0 else -1),
                }
            )

    def update(self):
        self.t += 0.018
        self.hue_offset += 0.003
        self.angle_y += 0.006
        self.angle_z += 0.0025

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.16
            dist = 6.5
            edges = []
            projected_rings = []

            core = rotate_3d(
                0.18 * math.sin(self.t * 1.3),
                0.16 * math.cos(self.t * 1.1),
                0.22 * math.sin(self.t * 1.7),
                self.angle_x,
                self.angle_y,
                self.angle_z,
            )
            core_proj = project_3d(*core, cx, cy, size, dist)

            for ring_idx, ring in enumerate(self.rings):
                ring_points = []
                for node_idx in range(self.nodes_per_ring):
                    a = (
                        ring["phase"]
                        + node_idx * 2 * math.pi / self.nodes_per_ring
                        + self.t * ring["speed"]
                    )
                    radial = ring["radius"] * (
                        1 + 0.04 * math.sin(self.t * 2.1 + node_idx * 0.55 + ring_idx)
                    )
                    x = radial * math.cos(a)
                    y = 0.22 * math.sin(2 * a + ring_idx * 0.8 + self.t * 1.5)
                    z = radial * math.sin(a)
                    x, y, z = rotate_3d(x, y, z, ring["tilt_x"], 0.0, ring["tilt_z"])
                    x, y, z = rotate_3d(
                        x, y, z, self.angle_x, self.angle_y, self.angle_z
                    )
                    ring_points.append(project_3d(x, y, z, cx, cy, size, dist))
                projected_rings.append(ring_points)

            for ring_idx, ring_points in enumerate(projected_rings):
                for node_idx in range(self.nodes_per_ring):
                    p1 = ring_points[node_idx]
                    p2 = ring_points[(node_idx + 1) % self.nodes_per_ring]
                    hue = (
                        self.hue_offset
                        + ring_idx * 0.1
                        + node_idx / self.nodes_per_ring * 0.2
                    ) % 1.0
                    depth = (p1[2] + p2[2] + 5.0) / 10.0
                    lw = max(1, int(2.4 * (p1[3] + p2[3]) / 2))
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.78, 0.35 + depth * 0.6),
                            lw,
                        )
                    )

                if ring_idx < len(projected_rings) - 1:
                    next_ring = projected_rings[ring_idx + 1]
                    shift = int((self.t * 12 + ring_idx * 3) % self.nodes_per_ring)
                    for node_idx in range(self.nodes_per_ring):
                        p1 = ring_points[node_idx]
                        p2 = next_ring[(node_idx + shift) % self.nodes_per_ring]
                        hue = (
                            self.hue_offset
                            + 0.45
                            + ring_idx * 0.07
                            + node_idx / self.nodes_per_ring * 0.08
                        ) % 1.0
                        depth = (p1[2] + p2[2] + 5.0) / 10.0
                        edges.append(
                            (
                                p1[0],
                                p1[1],
                                p2[0],
                                p2[1],
                                hsv_to_hex(hue, 0.55, 0.25 + depth * 0.45),
                                1,
                            )
                        )

                if ring_idx % 2 == 0:
                    for node_idx in range(0, self.nodes_per_ring, 3):
                        p1 = ring_points[node_idx]
                        hue = (self.hue_offset + 0.7 + ring_idx * 0.05) % 1.0
                        depth = (p1[2] + core_proj[2] + 5.0) / 10.0
                        edges.append(
                            (
                                p1[0],
                                p1[1],
                                core_proj[0],
                                core_proj[1],
                                hsv_to_hex(hue, 0.45, 0.2 + depth * 0.4),
                                1,
                            )
                        )

            return edges
        except Exception:
            return []


class MobiusRibbon:
    """Twisting Mobius ribbon rendered as a 3D wireframe band."""

    name = "Mobius Ribbon"
    num_lines = 430

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = 0.65
        self.angle_y = 0.0
        self.angle_z = 0.25
        self.segments = 110

    def update(self):
        self.t += 0.017
        self.hue_offset += 0.003
        self.angle_y += 0.011
        self.angle_x = 0.62 + 0.08 * math.sin(self.t * 0.6)
        self.angle_z += 0.0035

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.17
            dist = 6.0
            rails = [[], []]
            edges = []

            for i in range(self.segments + 1):
                u = i * 2 * math.pi / self.segments
                radius = 1.55 + 0.18 * math.sin(2 * u + self.t * 1.3)
                band_w = 0.44 + 0.06 * math.sin(4 * u - self.t * 2.2)
                ripple = 0.16 * math.sin(3 * u - self.t * 2.0)
                for rail_idx, side in enumerate((-1.0, 1.0)):
                    v = side * band_w
                    x = (radius + 0.55 * v * math.cos(u * 0.5)) * math.cos(u)
                    y = 0.82 * v * math.sin(u * 0.5) + 0.18 * math.sin(
                        2 * u + self.t * 1.6
                    )
                    z = (radius + 0.55 * v * math.cos(u * 0.5)) * math.sin(u) + ripple
                    x, y, z = rotate_3d(
                        x, y, z, self.angle_x, self.angle_y, self.angle_z
                    )
                    rails[rail_idx].append(project_3d(x, y, z, cx, cy, size, dist))

            for i in range(self.segments):
                for rail_idx, rail_points in enumerate(rails):
                    p1 = rail_points[i]
                    p2 = rail_points[i + 1]
                    hue = (
                        self.hue_offset + i / self.segments * 0.35 + rail_idx * 0.08
                    ) % 1.0
                    depth = (p1[2] + p2[2] + 5.0) / 10.0
                    lw = max(1, int(2.8 * (p1[3] + p2[3]) / 2))
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.8, 0.35 + depth * 0.55),
                            lw,
                        )
                    )

                left = rails[0][i]
                right = rails[1][i]
                hue = (self.hue_offset + 0.55 + i / self.segments * 0.2) % 1.0
                depth = (left[2] + right[2] + 5.0) / 10.0
                edges.append(
                    (
                        left[0],
                        left[1],
                        right[0],
                        right[1],
                        hsv_to_hex(hue, 0.5, 0.25 + depth * 0.45),
                        1,
                    )
                )

                if i % 2 == 0:
                    diag_a = rails[0][i]
                    diag_b = rails[1][i + 1]
                    hue = (self.hue_offset + 0.82 + i / self.segments * 0.1) % 1.0
                    edges.append(
                        (
                            diag_a[0],
                            diag_a[1],
                            diag_b[0],
                            diag_b[1],
                            hsv_to_hex(hue, 0.45, 0.32),
                            1,
                        )
                    )

            return edges
        except Exception:
            return []


class ToroidalReactor:
    """Animated torus grid with a glowing inner reactor ring."""

    name = "Toroidal Reactor"
    num_lines = 760

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = 0.9
        self.angle_y = 0.0
        self.angle_z = 0.14
        self.u_count = 26
        self.v_count = 12

    def update(self):
        self.t += 0.02
        self.hue_offset += 0.0035
        self.angle_y += 0.009
        self.angle_z += 0.002

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.16
            dist = 7.0
            grid = []
            edges = []

            for ui in range(self.u_count):
                u = ui * 2 * math.pi / self.u_count
                ring_wave = 0.18 * math.sin(3 * u + self.t * 1.4)
                row = []
                for vi in range(self.v_count):
                    v = vi * 2 * math.pi / self.v_count
                    minor = 0.48 + 0.08 * math.sin(2 * v - self.t * 2.0 + ui * 0.3)
                    radial = 1.65 + ring_wave + minor * math.cos(v)
                    x = radial * math.cos(u)
                    y = 0.95 * minor * math.sin(v) + 0.18 * math.sin(
                        2 * u + self.t * 1.1
                    )
                    z = radial * math.sin(u)
                    x, y, z = rotate_3d(
                        x, y, z, self.angle_x, self.angle_y, self.angle_z
                    )
                    row.append(project_3d(x, y, z, cx, cy, size, dist))
                grid.append(row)

            for ui in range(self.u_count):
                for vi in range(self.v_count):
                    p1 = grid[ui][vi]
                    p2 = grid[(ui + 1) % self.u_count][vi]
                    p3 = grid[ui][(vi + 1) % self.v_count]
                    depth12 = (p1[2] + p2[2] + 6.0) / 12.0
                    depth13 = (p1[2] + p3[2] + 6.0) / 12.0
                    hue_u = (
                        self.hue_offset
                        + ui / self.u_count * 0.5
                        + vi / self.v_count * 0.1
                    ) % 1.0
                    hue_v = (self.hue_offset + 0.35 + vi / self.v_count * 0.25) % 1.0
                    lw_u = max(1, int(2.7 * (p1[3] + p2[3]) / 2))
                    lw_v = max(1, int(2.2 * (p1[3] + p3[3]) / 2))
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue_u, 0.78, 0.3 + depth12 * 0.65),
                            lw_u,
                        )
                    )
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p3[0],
                            p3[1],
                            hsv_to_hex(hue_v, 0.55, 0.22 + depth13 * 0.48),
                            lw_v,
                        )
                    )

            core_ring = []
            inner_vi = self.v_count // 2
            for ui in range(self.u_count):
                u = ui * 2 * math.pi / self.u_count + self.t * 0.8
                x = 0.62 * math.cos(u)
                y = 0.15 * math.sin(3 * u + self.t * 2.0)
                z = 0.62 * math.sin(u)
                x, y, z = rotate_3d(x, y, z, self.angle_x, self.angle_y, self.angle_z)
                core_ring.append(project_3d(x, y, z, cx, cy, size, dist))

            for ui in range(self.u_count):
                p1 = core_ring[ui]
                p2 = core_ring[(ui + 1) % self.u_count]
                hue = (self.hue_offset + 0.82 + ui / self.u_count * 0.18) % 1.0
                depth = (p1[2] + p2[2] + 6.0) / 12.0
                edges.append(
                    (
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                        hsv_to_hex(hue, 0.95, 0.5 + depth * 0.45),
                        2,
                    )
                )

                if ui % 2 == 0:
                    shell = grid[ui][inner_vi]
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            shell[0],
                            shell[1],
                            hsv_to_hex((hue + 0.1) % 1.0, 0.55, 0.35),
                            1,
                        )
                    )

            return edges
        except Exception:
            return []


class CrystalMatrix:
    """Breathing cube lattice with depth-shaded crystalline diagonals."""

    name = "Crystal Matrix"
    num_lines = 240

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = 0.7
        self.angle_y = 0.0
        self.angle_z = 0.2
        self.coords = [-1.5, -0.5, 0.5, 1.5]

    def update(self):
        self.t += 0.02
        self.hue_offset += 0.003
        self.angle_y += 0.01
        self.angle_z += 0.003

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.135
            dist = 7.0
            points = {}
            edges = []

            for ix, x0 in enumerate(self.coords):
                for iy, y0 in enumerate(self.coords):
                    for iz, z0 in enumerate(self.coords):
                        x = x0 + 0.22 * math.sin(self.t * 1.2 + y0 * 1.6 + z0 * 0.9)
                        y = y0 + 0.22 * math.cos(self.t * 1.05 + x0 * 1.4 - z0 * 0.7)
                        z = z0 + 0.22 * math.sin(self.t * 1.35 + x0 * 0.8 + y0 * 1.1)
                        breathe = 1.0 + 0.05 * math.sin(self.t * 1.8 + x0 + y0 + z0)
                        x, y, z = x * breathe, y * breathe, z * breathe
                        x, y, z = rotate_3d(
                            x, y, z, self.angle_x, self.angle_y, self.angle_z
                        )
                        points[(ix, iy, iz)] = project_3d(x, y, z, cx, cy, size, dist)

            for ix in range(4):
                for iy in range(4):
                    for iz in range(4):
                        p = points[(ix, iy, iz)]
                        for dx, dy, dz, hue_shift in (
                            (1, 0, 0, 0.0),
                            (0, 1, 0, 0.18),
                            (0, 0, 1, 0.36),
                        ):
                            nx, ny, nz = ix + dx, iy + dy, iz + dz
                            if nx < 4 and ny < 4 and nz < 4:
                                q = points[(nx, ny, nz)]
                                hue = (
                                    self.hue_offset
                                    + hue_shift
                                    + ix * 0.05
                                    + iy * 0.03
                                    + iz * 0.02
                                ) % 1.0
                                depth = (p[2] + q[2] + 6.0) / 12.0
                                lw = max(1, int(2.4 * (p[3] + q[3]) / 2))
                                edges.append(
                                    (
                                        p[0],
                                        p[1],
                                        q[0],
                                        q[1],
                                        hsv_to_hex(hue, 0.75, 0.3 + depth * 0.6),
                                        lw,
                                    )
                                )

                        if (
                            iz in (0, 3)
                            and ix < 3
                            and iy < 3
                            and (ix + iy + iz) % 2 == 0
                        ):
                            q = points[(ix + 1, iy + 1, iz)]
                            hue = (self.hue_offset + 0.62 + iz * 0.05) % 1.0
                            depth = (p[2] + q[2] + 6.0) / 12.0
                            edges.append(
                                (
                                    p[0],
                                    p[1],
                                    q[0],
                                    q[1],
                                    hsv_to_hex(hue, 0.45, 0.24 + depth * 0.42),
                                    1,
                                )
                            )

            return edges
        except Exception:
            return []


class AizawaAttractor:
    """Dense 3D strange attractor with rotating depth shimmer."""

    name = "Aizawa Attractor"
    num_lines = 900

    def __init__(self):
        self.hue_offset = 0.0
        self.angle_x = 0.72
        self.angle_y = 0.0
        self.angle_z = 0.18
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.points = []
        for _ in range(1400):
            self._step(0.01, False)
        for _ in range(700):
            self._step(0.01, True)

    def _step(self, dt, store=True):
        a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
        x, y, z = self.x, self.y, self.z
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - (z**3) / 3 - (x * x + y * y) * (1 + e * z) + f * z * (x**3)
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        if store:
            self.points.append((self.x, self.y, self.z))
            if len(self.points) > 860:
                del self.points[: len(self.points) - 860]

    def update(self):
        for _ in range(8):
            self._step(0.01, True)
        self.hue_offset += 0.003
        self.angle_y += 0.006
        self.angle_z += 0.0018
        self.angle_x = 0.68 + 0.08 * math.sin(self.hue_offset * 40)

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.22
            dist = 7.0
            projected = []
            edges = []

            for x, y, z in self.points:
                px = x * 0.34
                py = y * 0.34
                pz = (z - 0.75) * 0.34
                px, py, pz = rotate_3d(
                    px, py, pz, self.angle_x, self.angle_y, self.angle_z
                )
                projected.append(project_3d(px, py, pz, cx, cy, size, dist))

            for idx in range(1, len(projected)):
                p1 = projected[idx - 1]
                p2 = projected[idx]
                trail = idx / max(1, len(projected) - 1)
                hue = (self.hue_offset + 0.72 + trail * 0.2) % 1.0
                depth = (p1[2] + p2[2] + 6.0) / 12.0
                value = max(0.08, min(1.0, 0.08 + trail * 0.65 + depth * 0.22))
                lw = 2 if trail > 0.82 else 1
                edges.append(
                    (p1[0], p1[1], p2[0], p2[1], hsv_to_hex(hue, 0.82, value), lw)
                )

            return edges
        except Exception:
            return []


class QuantumTunnel:
    """Warped polygonal tunnel with moire-like depth links."""

    name = "Quantum Tunnel"
    num_lines = 620

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.sides = 10
        self.rings = 20

    def update(self):
        self.t += 0.03
        self.hue_offset += 0.004

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.28
            dist = 6.5
            ring_points = []
            edges = []

            for ring_idx in range(self.rings):
                depth = ring_idx / max(1, self.rings - 1)
                z = -2.8 + depth * 5.4
                drift_x = 0.45 * math.sin(self.t * 0.7 + depth * 1.8)
                drift_y = 0.32 * math.cos(self.t * 0.9 + depth * 2.1)
                drift_x += 0.12 * depth * math.sin(self.t + depth * 5)
                drift_y += 0.10 * depth * math.cos(self.t * 1.3 + depth * 4)
                twist = self.t * 1.1 + depth * 2.8
                base_r = 0.65 + 0.22 * math.sin(self.t * 1.8 + depth * 6.0)
                ring = []

                for side_idx in range(self.sides):
                    a = side_idx * 2 * math.pi / self.sides + twist
                    rr = base_r * (
                        1 + 0.18 * math.sin(3 * a - self.t * 2.2 + depth * 4.0)
                    )
                    x = drift_x + rr * math.cos(a)
                    y = drift_y + 0.9 * rr * math.sin(a)
                    ring.append(project_3d(x, y, z, cx, cy, size, dist))

                ring_points.append(ring)

            for ring_idx, ring in enumerate(ring_points):
                for side_idx in range(self.sides):
                    p1 = ring[side_idx]
                    p2 = ring[(side_idx + 1) % self.sides]
                    hue = (
                        self.hue_offset
                        + ring_idx / self.rings * 0.45
                        + side_idx / self.sides * 0.08
                    ) % 1.0
                    depth = (p1[2] + p2[2] + 6.0) / 12.0
                    lw = max(1, int(2.5 * (p1[3] + p2[3]) / 2))
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.8, 0.25 + depth * 0.65),
                            lw,
                        )
                    )

                if ring_idx < self.rings - 1:
                    next_ring = ring_points[ring_idx + 1]
                    for side_idx in range(self.sides):
                        p1 = ring[side_idx]
                        p2 = next_ring[side_idx]
                        p3 = next_ring[(side_idx + 1) % self.sides]
                        hue = (
                            self.hue_offset + 0.5 + side_idx / self.sides * 0.1
                        ) % 1.0
                        edges.append(
                            (p1[0], p1[1], p2[0], p2[1], hsv_to_hex(hue, 0.45, 0.3), 1)
                        )
                        if ring_idx % 2 == 0:
                            edges.append(
                                (
                                    p1[0],
                                    p1[1],
                                    p3[0],
                                    p3[1],
                                    hsv_to_hex((hue + 0.08) % 1.0, 0.35, 0.24),
                                    1,
                                )
                            )

            return edges
        except Exception:
            return []


class EventHorizon:
    """Accretion rings and infalling streamlines around a tilted core."""

    name = "Event Horizon"
    num_lines = 520

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.precession = 0.0
        self.tilt_x = 1.1
        self.tilt_z = 0.18
        self.ring_count = 7
        self.seg_count = 56

    def update(self):
        self.t += 0.02
        self.hue_offset += 0.0035
        self.precession += 0.012
        self.tilt_x = 1.08 + 0.08 * math.sin(self.t * 0.7)
        self.tilt_z += 0.002

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.19
            dist = 7.0
            edges = []

            for ring_idx in range(self.ring_count):
                base_r = 0.55 + ring_idx * 0.18
                prev = None
                for seg_idx in range(self.seg_count + 1):
                    a = seg_idx * 2 * math.pi / self.seg_count
                    twist = self.precession + ring_idx * 0.14
                    r = base_r * (1 + 0.04 * math.sin(4 * a - self.t * 2.0 + ring_idx))
                    x = r * math.cos(a + twist)
                    y = 0.07 * ring_idx * math.sin(2 * a + self.t * 1.3)
                    z = 0.68 * r * math.sin(a + twist)
                    x, y, z = rotate_3d(x, y, z, self.tilt_x, 0.0, self.tilt_z)
                    cur = project_3d(x, y, z, cx, cy, size, dist)
                    if prev is not None:
                        hue = (
                            self.hue_offset
                            + 0.06 * ring_idx
                            + seg_idx / self.seg_count * 0.08
                        ) % 1.0
                        depth = (prev[2] + cur[2] + 6.0) / 12.0
                        lw = 2 if ring_idx >= self.ring_count - 2 else 1
                        edges.append(
                            (
                                prev[0],
                                prev[1],
                                cur[0],
                                cur[1],
                                hsv_to_hex(hue, 0.78, 0.2 + depth * 0.55),
                                lw,
                            )
                        )
                    prev = cur

            streams = 6
            steps = 34
            for stream_idx in range(streams):
                prev = None
                phase = stream_idx * 2 * math.pi / streams + self.t * 1.1
                for step_idx in range(steps + 1):
                    frac = step_idx / steps
                    r = 2.0 - frac * 1.65
                    a = phase + frac * 5.5
                    x = r * math.cos(a)
                    y = 0.14 * math.sin(frac * 8 + self.t * 2.0 + stream_idx)
                    z = 0.75 * r * math.sin(a) + frac * 0.55
                    x, y, z = rotate_3d(x, y, z, self.tilt_x, 0.0, self.tilt_z)
                    cur = project_3d(x, y, z, cx, cy, size, dist)
                    if prev is not None:
                        hue = (
                            self.hue_offset + 0.72 + stream_idx * 0.05 + frac * 0.08
                        ) % 1.0
                        edges.append(
                            (
                                prev[0],
                                prev[1],
                                cur[0],
                                cur[1],
                                hsv_to_hex(hue, 0.65, 0.25 + frac * 0.45),
                                1,
                            )
                        )
                    prev = cur

            core = []
            core_radius = 0.24
            for seg_idx in range(18):
                a = seg_idx * 2 * math.pi / 18 + self.precession * 1.4
                x = core_radius * math.cos(a)
                y = 0.02 * math.sin(3 * a + self.t * 2.5)
                z = 0.68 * core_radius * math.sin(a)
                x, y, z = rotate_3d(x, y, z, self.tilt_x, 0.0, self.tilt_z)
                core.append(project_3d(x, y, z, cx, cy, size, dist))

            for seg_idx in range(len(core)):
                p1 = core[seg_idx]
                p2 = core[(seg_idx + 1) % len(core)]
                hue = (self.hue_offset + 0.12 + seg_idx / len(core) * 0.12) % 1.0
                edges.append(
                    (p1[0], p1[1], p2[0], p2[1], hsv_to_hex(hue, 0.95, 0.75), 2)
                )

            return edges
        except Exception:
            return []


class PrismaticStorm:
    """Orbiting octahedral shards with trails and lightning-like cross-links."""

    name = "Prismatic Storm"
    num_lines = 420

    _OCTA_VERTS = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1.25),
        (0, 0, -1.25),
    ]
    _OCTA_EDGES = [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]

    def __init__(self):
        self.t = 0.0
        self.hue_offset = 0.0
        self.angle_x = 0.72
        self.angle_y = 0.0
        self.angle_z = 0.14
        self.shards = []
        self.trails = []
        rng = random.Random(108)
        for idx in range(14):
            self.shards.append(
                {
                    "orbit_r": rng.uniform(1.0, 2.2),
                    "orbit_phase": rng.random() * 2 * math.pi,
                    "orbit_speed": rng.uniform(0.45, 1.1) * (1 if idx % 2 == 0 else -1),
                    "vertical_phase": rng.random() * 2 * math.pi,
                    "size": rng.uniform(0.16, 0.32),
                    "spin_x": rng.uniform(-1.2, 1.2),
                    "spin_y": rng.uniform(-1.4, 1.4),
                    "spin_z": rng.uniform(-1.1, 1.1),
                    "spin_speed": rng.uniform(0.8, 1.8) * (1 if idx % 3 else -1),
                }
            )
            self.trails.append([])

    def _center_for(self, shard, idx):
        orbit = shard["orbit_phase"] + self.t * shard["orbit_speed"]
        radius = shard["orbit_r"] * (1.0 + 0.16 * math.sin(self.t * 1.7 + idx * 0.45))
        x = radius * math.cos(orbit)
        y = 0.65 * math.sin(self.t * 1.1 + shard["vertical_phase"])
        y += 0.22 * math.sin(2.4 * orbit + idx * 0.3)
        z = radius * math.sin(orbit)
        x += 0.18 * math.sin(self.t * 0.9 + idx * 0.8)
        z += 0.18 * math.cos(self.t * 1.0 + idx * 0.6)
        return x, y, z

    def update(self):
        self.t += 0.02
        self.hue_offset += 0.0035
        self.angle_y += 0.01
        self.angle_z += 0.0025

        for idx, shard in enumerate(self.shards):
            self.trails[idx].append(self._center_for(shard, idx))
            if len(self.trails[idx]) > 9:
                self.trails[idx] = self.trails[idx][-9:]

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            size = min(width, height) * 0.17
            dist = 7.0
            edges = []
            centers = []

            core_x = 0.18 * math.sin(self.t * 1.7)
            core_y = 0.12 * math.cos(self.t * 1.3)
            core_z = 0.18 * math.sin(self.t * 2.2)
            core_proj = project_3d(
                *rotate_3d(
                    core_x, core_y, core_z, self.angle_x, self.angle_y, self.angle_z
                ),
                cx,
                cy,
                size,
                dist,
            )

            for idx, shard in enumerate(self.shards):
                center = self._center_for(shard, idx)
                spin = self.t * shard["spin_speed"]
                scale = shard["size"] * (
                    1.0 + 0.28 * math.sin(self.t * 2.3 + idx * 0.6)
                )
                shard_points = []

                for vx, vy, vz in self._OCTA_VERTS:
                    px = vx * scale
                    py = vy * scale
                    pz = vz * scale
                    px, py, pz = rotate_3d(
                        px,
                        py,
                        pz,
                        shard["spin_x"] + spin,
                        shard["spin_y"] - spin * 0.8,
                        shard["spin_z"] + spin * 1.15,
                    )
                    px += center[0]
                    py += center[1]
                    pz += center[2]
                    px, py, pz = rotate_3d(
                        px, py, pz, self.angle_x, self.angle_y, self.angle_z
                    )
                    shard_points.append(project_3d(px, py, pz, cx, cy, size, dist))

                center_proj = project_3d(
                    *rotate_3d(
                        center[0],
                        center[1],
                        center[2],
                        self.angle_x,
                        self.angle_y,
                        self.angle_z,
                    ),
                    cx,
                    cy,
                    size,
                    dist,
                )
                centers.append(center_proj)

                for edge_idx, (a, b) in enumerate(self._OCTA_EDGES):
                    p1 = shard_points[a]
                    p2 = shard_points[b]
                    hue = (
                        self.hue_offset
                        + idx * 0.055
                        + edge_idx / len(self._OCTA_EDGES) * 0.12
                    ) % 1.0
                    depth = (p1[2] + p2[2] + 6.0) / 12.0
                    lw = max(1, int(2.6 * (p1[3] + p2[3]) / 2))
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.82, 0.28 + depth * 0.62),
                            lw,
                        )
                    )

                trail = self.trails[idx]
                for t_idx in range(1, len(trail)):
                    x1, y1, z1 = rotate_3d(
                        *trail[t_idx - 1], self.angle_x, self.angle_y, self.angle_z
                    )
                    x2, y2, z2 = rotate_3d(
                        *trail[t_idx], self.angle_x, self.angle_y, self.angle_z
                    )
                    p1 = project_3d(x1, y1, z1, cx, cy, size, dist)
                    p2 = project_3d(x2, y2, z2, cx, cy, size, dist)
                    fade = t_idx / max(1, len(trail) - 1)
                    hue = (self.hue_offset + 0.68 + idx * 0.03) % 1.0
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.45, 0.14 + fade * 0.35),
                            1,
                        )
                    )

                if idx % 2 == 0:
                    hue = (self.hue_offset + 0.88 + idx * 0.02) % 1.0
                    depth = (center_proj[2] + core_proj[2] + 6.0) / 12.0
                    edges.append(
                        (
                            center_proj[0],
                            center_proj[1],
                            core_proj[0],
                            core_proj[1],
                            hsv_to_hex(hue, 0.55, 0.18 + depth * 0.35),
                            1,
                        )
                    )

            for idx, p1 in enumerate(centers):
                p2 = centers[(idx + 3) % len(centers)]
                p3 = centers[(idx + 7) % len(centers)]
                hue = (self.hue_offset + 0.42 + idx * 0.025) % 1.0
                edges.append(
                    (p1[0], p1[1], p2[0], p2[1], hsv_to_hex(hue, 0.3, 0.22), 1)
                )
                if idx % 3 == 0:
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p3[0],
                            p3[1],
                            hsv_to_hex((hue + 0.08) % 1.0, 0.24, 0.18),
                            1,
                        )
                    )

            return edges
        except Exception:
            return []


class HexagonalMesh:
    """Tessellated hexagonal mesh with wave-displaced shared vertices."""

    name = "Hexagonal Mesh"
    num_lines = 1700

    def __init__(self):
        self.t = 0.0
        self.wave_angle = random.random() * math.pi * 2
        self.next_change = 200 + random.random() * 200
        self.hue_offset = 0.0
        self.params = {
            "wave_speed": 0.02 + random.random() * 0.015,
            "wave_freq": 0.05 + random.random() * 0.04,
            "vert_amp": 0.15,
            "hue_speed": 0.004 + random.random() * 0.003,
        }

    def update(self):
        p = self.params
        self.t += p["wave_speed"]
        self.hue_offset += p["hue_speed"]
        if self.t > self.next_change:
            self.wave_angle += math.pi / 3 + random.random() * math.pi / 3
            self.next_change = self.t + 180 + random.random() * 220

    def get_edges(self, width, height):
        try:
            edges = []
            sz = min(width, height) * 0.07
            sqrt3 = math.sqrt(3)
            dx, dy = sz * 1.5, sz * sqrt3
            cols = int(width / dx) + 2
            rows = int(height / dy) + 2
            p = self.params
            wcos, wsin = math.cos(self.wave_angle), math.sin(self.wave_angle)

            verts = {}
            for row in range(-1, rows + 1):
                for col in range(-1, cols + 1):
                    hcx = col * dx + sz * 0.5
                    hcy = row * dy + sz * 0.5
                    if col % 2:
                        hcy += dy / 2

                    for idx in range(6):
                        a = idx * math.pi / 3 + math.pi / 6
                        vx = hcx + sz * math.cos(a)
                        vy = hcy + sz * math.sin(a)
                        key = (round(vx, 1), round(vy, 1))
                        if key in verts:
                            continue

                        dist = vx * wcos + vy * wsin
                        phase = dist * p["wave_freq"]
                        wave = math.sin(self.t - phase)
                        dcx, dcy = vx - width / 2, vy - height / 2
                        d = math.hypot(dcx, dcy) + 0.001
                        disp = wave * sz * p["vert_amp"]
                        verts[key] = (vx + dcx / d * disp, vy + dcy / d * disp, dist)

            drawn = set()
            for row in range(rows):
                for col in range(cols):
                    hcx = col * dx + sz * 0.5
                    hcy = row * dy + sz * 0.5
                    if col % 2:
                        hcy += dy / 2
                    if hcx > width + sz or hcy > height + sz:
                        continue

                    hex_verts = []
                    for idx in range(6):
                        a = idx * math.pi / 3 + math.pi / 6
                        vx = hcx + sz * math.cos(a)
                        vy = hcy + sz * math.sin(a)
                        key = (round(vx, 1), round(vy, 1))
                        hex_verts.append(verts.get(key, (vx, vy, 0)))

                    for idx in range(6):
                        v1, v2 = hex_verts[idx], hex_verts[(idx + 1) % 6]
                        edge_key = (min(v1[:2], v2[:2]), max(v1[:2], v2[:2]))
                        if edge_key in drawn:
                            continue
                        drawn.add(edge_key)
                        hue = ((v1[2] + v2[2]) * 0.002 + self.hue_offset) % 1.0
                        edges.append(
                            (
                                v1[0],
                                v1[1],
                                v2[0],
                                v2[1],
                                hsv_to_hex(hue, 0.7, 0.6),
                                2,
                            )
                        )

                    hue2 = (hcx * 0.003 + self.hue_offset + 0.35) % 1.0
                    for idx, vertex in enumerate(hex_verts):
                        edges.append(
                            (
                                hcx,
                                hcy,
                                vertex[0],
                                vertex[1],
                                hsv_to_hex((hue2 + idx * 0.1) % 1, 0.85, 0.7),
                                2,
                            )
                        )

            return edges
        except Exception:
            return []


class HyperOrbits:
    """4D particle swarm rendered as luminous crosses and orbit links."""

    name = "HyperOrbits"
    num_lines = 900

    def __init__(self, num_points=350):
        self.num_points = num_points
        self.start_time = time.time()
        self.randomize_parameters()
        self._create_particles()

    def randomize_parameters(self):
        self.params = {
            "global_speed": 0.5 + random.random() * 1.0,
            "rotation_speed": 0.1 + random.random() * 0.3,
            "p1_freq": 0.1 + random.random() * 0.15,
            "p2_freq": 0.08 + random.random() * 0.1,
            "p3_freq": 0.05 + random.random() * 0.08,
            "p4_freq": 0.12 + random.random() * 0.15,
            "angle_mix": 0.5 + random.random() * 0.5,
            "angle2_mix": 0.4 + random.random() * 0.6,
            "z_influence": 0.3 + random.random() * 0.7,
            "hue_speed": 0.1 + random.random() * 0.2,
            "hue_z_mix": 0.08 + random.random() * 0.15,
            "hue_radius_mix": 0.2 + random.random() * 0.3,
            "saturation_base": 0.5 + random.random() * 0.3,
            "value_base": 0.2 + random.random() * 0.2,
            "size_breath": 0.4 + random.random() * 0.5,
            "base_scale": 0.3 + random.random() * 0.15,
        }
        self.start_time = time.time()

    def _create_particles(self):
        self.particles = []
        for _ in range(self.num_points):
            self.particles.append(
                {
                    "theta": random.random() * 2 * math.pi,
                    "radius": 0.15 + 0.85 * (random.random() ** 1.5),
                    "phase": random.random() * 2 * math.pi,
                    "speed": 0.3 + random.random() * 2.0,
                    "size": 1.0 + 2.5 * (random.random() ** 0.5),
                    "hue_offset": random.random(),
                }
            )

    def update(self):
        pass

    def _get_points(self, width, height):
        t = (time.time() - self.start_time) * self.params["global_speed"]
        cx, cy = width / 2, height / 2
        base_scale = min(width, height) * self.params["base_scale"]

        p1 = math.sin(t * self.params["p1_freq"])
        p2 = math.sin(t * self.params["p2_freq"] + 1.3)
        p3 = math.sin(t * self.params["p3_freq"] + 2.7)
        p4 = math.sin(t * self.params["p4_freq"] + 0.4)

        points = []
        for particle in self.particles:
            local_t = (
                t * particle["speed"] * self.params["rotation_speed"]
                + particle["phase"]
            )
            angle = (
                particle["theta"]
                + local_t * (0.3 + self.params["angle_mix"] * p1)
                + particle["radius"] * (0.6 * p2)
            )
            angle2 = (
                particle["theta"] * (1.5 + 0.8 * p3)
                + local_t * (0.8 + self.params["angle2_mix"] * p4)
            )
            x_norm = math.sin(angle) + self.params["angle_mix"] * math.sin(angle2)
            y_norm = math.cos(angle) + self.params["angle_mix"] * math.cos(
                angle2 + 1.7
            )
            z = math.sin(angle * 0.5 + local_t * self.params["z_influence"])
            scale = base_scale * particle["radius"]
            size = particle["size"] * (
                0.8 + self.params["size_breath"] * (0.5 + 0.5 * z)
            )
            hue = (
                self.params["hue_z_mix"] * z
                + self.params["hue_radius_mix"] * particle["radius"]
                + self.params["hue_speed"] * math.sin(local_t * 0.3)
                + particle["hue_offset"]
            ) % 1.0
            sat = self.params["saturation_base"] + 0.35 * abs(z)
            val = self.params["value_base"] + 0.7 * abs(math.sin(angle))
            points.append(
                (
                    cx + x_norm * scale,
                    cy + y_norm * scale,
                    size,
                    hsv_to_hex(hue, min(1, sat), min(1, val)),
                )
            )
        return points

    def get_edges(self, width, height):
        try:
            points = self._get_points(width, height)
            edges = []
            for idx, (x, y, size, color) in enumerate(points):
                half = max(1.0, size * 0.65)
                edges.append((x - half, y, x + half, y, color, 1))
                edges.append((x, y - half, x, y + half, color, 1))
                if idx % 12 == 0:
                    x2, y2, _, color2 = points[(idx + 29) % len(points)]
                    edges.append((x, y, x2, y2, color2, 1))
            return edges
        except Exception:
            return []


class ModularResonance:
    """Modular arithmetic chords with irrational multiplier modulation."""

    name = "Modular Resonance"
    num_lines = 120

    def __init__(self):
        self.n_points = 92
        self.t = 0.0
        self.randomize_parameters()
        self.angles = [2 * math.pi * idx / self.n_points for idx in range(self.n_points)]
        self.cos_a = [math.cos(angle) for angle in self.angles]
        self.sin_a = [math.sin(angle) for angle in self.angles]

    def randomize_parameters(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.psi = self.phi - 1
        self.params = {
            "base_mult": 2 + random.random() * 3,
            "wave_amp": 25 + random.random() * 20,
            "wave_speed": 0.06 + random.random() * 0.08,
            "phi_amp": 10 + random.random() * 15,
            "psi_amp": 5 + random.random() * 10,
            "sqrt2_amp": 3 + random.random() * 7,
            "breath_amp": 0.01 + random.random() * 0.02,
            "breath_speed": 1.0 + random.random() * 1.0,
            "hue_speed": 0.02 + random.random() * 0.04,
            "time_speed": 0.002 + random.random() * 0.002,
        }
        self.t = random.random() * 100

    def update(self):
        self.t += self.params["time_speed"]

    def get_edges(self, width, height):
        try:
            cx, cy = width / 2, height / 2
            base_radius = min(width, height) * 0.38
            p = self.params
            multiplier = (
                p["base_mult"]
                + p["wave_amp"] * (1 + math.sin(self.t * p["wave_speed"]))
                + p["phi_amp"] * math.sin(self.t * p["wave_speed"] * self.phi)
                + p["psi_amp"] * math.sin(self.t * p["wave_speed"] * self.psi)
                + p["sqrt2_amp"] * math.sin(self.t * p["wave_speed"] * math.sqrt(2))
            )
            breath = 1 + p["breath_amp"] * math.sin(self.t * p["breath_speed"])
            radius = base_radius * breath
            px = [cx + radius * c for c in self.cos_a]
            py = [cy + radius * s for s in self.sin_a]

            edges = []
            for idx in range(self.n_points):
                dest = int(idx * multiplier) % self.n_points
                if idx == dest:
                    continue
                x1, y1 = px[idx], py[idx]
                x2, y2 = px[dest], py[dest]
                hue = (idx / self.n_points + self.t * p["hue_speed"]) % 1.0
                jump = abs(dest - idx) / self.n_points
                dist = math.hypot(x2 - x1, y2 - y1)
                sat = 0.5 + 0.5 * jump
                val = 0.25 + 0.65 * min(1, dist / radius)
                edges.append((x1, y1, x2, y2, hsv_to_hex(hue, sat, val), 1))
            return edges
        except Exception:
            return []


class TriplePendulum:
    """Driven three-link pendulum with two elbow joints and fading trails."""

    name = "Triple Pendulum"
    num_lines = 520

    def __init__(self):
        self.t = 0.0
        self.hue_offset = random.random()
        self.lengths = [1.0, 0.82, 0.68]
        self.points = [
            [0.0, 0.0],
            [0.78, 0.65],
            [1.12, 1.42],
            [0.72, 2.02],
        ]
        self.prev_points = [point[:] for point in self.points]
        self.trails = [[], [], []]

    def _pivot(self):
        return [
            0.28 * math.sin(self.t * 0.83),
            -1.38 + 0.12 * math.cos(self.t * 0.57),
        ]

    def _satisfy_constraints(self):
        self.points[0] = self._pivot()
        for _ in range(7):
            self.points[0] = self._pivot()
            for idx, length in enumerate(self.lengths):
                p1 = self.points[idx]
                p2 = self.points[idx + 1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist = math.hypot(dx, dy) or 0.001
                diff = (dist - length) / dist
                if idx == 0:
                    p2[0] -= dx * diff
                    p2[1] -= dy * diff
                else:
                    correction = 0.5 * diff
                    p1[0] += dx * correction
                    p1[1] += dy * correction
                    p2[0] -= dx * correction
                    p2[1] -= dy * correction

    def update(self):
        self.t += 0.018
        self.hue_offset += 0.003
        gravity = 0.010
        damping = 0.996

        self.points[0] = self._pivot()
        for idx in range(1, 4):
            x, y = self.points[idx]
            px, py = self.prev_points[idx]
            vx = (x - px) * damping
            vy = (y - py) * damping + gravity
            drive = 0.0025 * math.sin(self.t * (1.3 + idx * 0.37) + idx)
            self.prev_points[idx] = [x, y]
            self.points[idx] = [x + vx + drive, y + vy]

        self._satisfy_constraints()
        for idx in range(3):
            bob = self.points[idx + 1]
            self.trails[idx].append((bob[0], bob[1]))
            if len(self.trails[idx]) > 58:
                self.trails[idx] = self.trails[idx][-58:]

    def _project(self, point, width, height):
        cx, cy = width / 2, height * 0.40
        scale = min(width, height) * 0.20
        return cx + point[0] * scale, cy + point[1] * scale

    def get_edges(self, width, height):
        try:
            edges = []
            projected = [self._project(point, width, height) for point in self.points]

            for idx in range(3):
                p1 = projected[idx]
                p2 = projected[idx + 1]
                hue = (self.hue_offset + idx * 0.11) % 1.0
                edges.append(
                    (
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                        hsv_to_hex(hue, 0.88, 0.8),
                        3 - min(idx, 1),
                    )
                )

            for trail_idx, trail in enumerate(self.trails):
                for point_idx in range(1, len(trail)):
                    p1 = self._project(trail[point_idx - 1], width, height)
                    p2 = self._project(trail[point_idx], width, height)
                    fade = point_idx / max(1, len(trail) - 1)
                    hue = (self.hue_offset + 0.55 + trail_idx * 0.08) % 1.0
                    edges.append(
                        (
                            p1[0],
                            p1[1],
                            p2[0],
                            p2[1],
                            hsv_to_hex(hue, 0.72, 0.12 + fade * 0.58),
                            1,
                        )
                    )

            for idx, point in enumerate(projected[1:], start=1):
                r = 4 + idx * 2
                hue = (self.hue_offset + 0.18 + idx * 0.13) % 1.0
                color = hsv_to_hex(hue, 0.95, 0.9)
                edges.extend(
                    [
                        (point[0] - r, point[1], point[0] + r, point[1], color, 2),
                        (point[0], point[1] - r, point[0], point[1] + r, color, 2),
                    ]
                )

            for idx in range(len(projected) - 2):
                p1 = projected[idx]
                p3 = projected[idx + 2]
                hue = (self.hue_offset + 0.78 + idx * 0.05) % 1.0
                edges.append(
                    (p1[0], p1[1], p3[0], p3[1], hsv_to_hex(hue, 0.32, 0.25), 1)
                )

            return edges
        except Exception:
            return []


# =============================================================================
# SCREENSAVER (MAIN)
# =============================================================================


class Screensaver:
    VISUALIZATIONS = [
        EventHorizon,
        ToroidalReactor,
        AizawaAttractor,
        QuantumTunnel,
        TriplePendulum,
        HexagonalMesh,
        MagneticField,
        MathJellyfish,
        MobiusRibbon,
        GeodesicSphere,
        WireframeTerrain,
        PrismaticStorm,
        HyperOrbits,
        KaleidoscopeMandala,
        CosmicWeb,
        ModularResonance,
        HilbertCurve,
        Lissajous3D,
        OrbitalLattice,
        SacredMandala,
        SriYantra,
        UzumakiSpiral3D,
        CreaturesViz,
        PendulumWave,
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spectral Geometry Screensaver 09")
        self.root.configure(bg=CONFIG["bg_color"])

        self.width, self.height = 900, 700
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.minsize(400, 300)

        # Clock bar
        self.clock_frame = tk.Frame(
            self.root, bg="#000000", height=CONFIG["clock_height"]
        )
        self.clock_frame.pack(fill=tk.X, side=tk.TOP)
        self.clock_frame.pack_propagate(False)
        self.clock_label = tk.Label(
            self.clock_frame,
            text="",
            font=("Arial", CONFIG["clock_font_size"], "bold"),
            fg="#ffffff",
            bg="#000000",
        )
        self.clock_label.pack(expand=True)
        self.name_label = tk.Label(
            self.clock_frame, text="", font=("Arial", 14), fg="#666666", bg="#000000"
        )
        self.name_label.place(relx=0.98, rely=0.5, anchor="e")

        # Canvas
        self.canvas = tk.Canvas(self.root, bg=CONFIG["bg_color"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Stars (always visible)
        self.stars = StarField(self.width, self.height)

        # Current visualization
        self.current_viz = None
        self.viz_index = -1
        self._switch_visualization()

        # Forward mouse motion to viz if it supports it
        self.canvas.bind("<Motion>", self._on_motion)

        # Pre-allocate canvas items (line pool)
        self.star_items = [
            self.canvas.create_oval(0, 0, 1, 1, fill="", outline="")
            for _ in range(CONFIG["num_stars"])
        ]
        self.line_items = [
            self.canvas.create_line(0, 0, 0, 0, fill="", width=1)
            for _ in range(CONFIG["max_lines"])
        ]
        self.line_index = 0

        # Bindings
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<space>", lambda e: self._switch_visualization())
        self.root.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.frame_time = 1000 // CONFIG["fps"]
        self.running = True

        self.root.after(CONFIG["switch_interval"], self._auto_switch)

    def _switch_visualization(self):
        self.viz_index = (self.viz_index + 1) % len(self.VISUALIZATIONS)
        VizClass = self.VISUALIZATIONS[self.viz_index]
        self.current_viz = VizClass()
        if hasattr(self.current_viz, "resize"):
            self.current_viz.resize(self.width, self.height)
        self.name_label.config(text=self.current_viz.name)
        self.root.title(f"Screensaver - {self.current_viz.name}")

    def _auto_switch(self):
        if self.running:
            self._switch_visualization()
            self.root.after(CONFIG["switch_interval"], self._auto_switch)

    def _on_motion(self, event):
        if hasattr(self.current_viz, "on_motion"):
            self.current_viz.on_motion(event)

    def _on_resize(self, event):
        if event.widget == self.canvas:
            self.width, self.height = event.width, event.height
            self.stars.resize(self.width, self.height)
            if hasattr(self.current_viz, "resize"):
                self.current_viz.resize(self.width, self.height)

    def quit(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def draw(self):
        try:
            self.clock_label.config(text=datetime.now().strftime("%H:%M:%S"))
            self.line_index = 0

            # Draw stars
            s = CONFIG["star_size"]
            for i, (x, y, color) in enumerate(self.stars.get_stars()):
                if i < len(self.star_items):
                    self.canvas.coords(self.star_items[i], x - s, y - s, x + s, y + s)
                    self.canvas.itemconfig(self.star_items[i], fill=color)

            # Draw current visualization
            if self.current_viz:
                for edge in self.current_viz.get_edges(self.width, self.height):
                    self._draw_line(edge)

            # Hide unused lines
            for i in range(self.line_index, len(self.line_items)):
                self.canvas.coords(self.line_items[i], 0, 0, 0, 0)
        except Exception:
            pass

    def _draw_line(self, edge):
        if self.line_index < len(self.line_items):
            x1, y1, x2, y2, color, lw = edge
            item = self.line_items[self.line_index]
            self.canvas.coords(item, x1, y1, x2, y2)
            self.canvas.itemconfig(item, fill=color, width=lw)
            self.line_index += 1

    def run(self):
        def frame():
            if not self.running:
                return
            try:
                self.stars.update()
                if self.current_viz:
                    self.current_viz.update()
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
