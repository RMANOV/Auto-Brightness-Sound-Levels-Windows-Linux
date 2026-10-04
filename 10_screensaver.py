#!/usr/bin/env python3
"""Spectral Geometry Screensaver 10 — Claude Edition — Pure Tkinter

Осем визуализации, които ги няма в 01-09:
  1. Hopf Fibration        — окръжностите на S3, стереографски в R3 (вложени тори)
  2. Three-Body Ballet     — гравитационната хореография "осмица" (Chenciner-Montgomery)
  3. Hyperbolic Flow       — геодезична мрежа в диска на Поанкаре под Мьобиус поток
  4. Clifford Attractor    — странен атрактор с морфиращи параметри
  5. Chaos Butterflies     — рояк двойни махала, разминаващи се от 1e-5 разлика
  6. Klein Bottle          — въртяща се неориентируема повърхнина (bagel immersion)
  7. Torus Knot Morph      — (p,q) възли, преливащи един в друг
  8. Murmuration           — boids ято като скорци привечер

Стил и архитектура: съвместими с 09 (CONFIG, StarField, line pool, часовник,
авто-превключване, ESC изход, SPACE следваща). Само stdlib: math/random/tkinter.

Selftest без екран: python 10_screensaver.py --selftest
"""

import math
import random
import sys
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
    "clock_font_size": 48,
    "line_width": 2,
    "max_lines": 3600,
    "switch_interval": 7 * 60 * 1000,  # 7 минути
}

TAU = 2 * math.pi


# =============================================================================
# UTILITY (стил 09)
# =============================================================================


_HSV_CACHE = {}


def hsv_to_hex(h, s, v):
    # квантуване (1/48 hue, 1/20 s+v): визуално идентично, но цветът на елемент
    # се СТАБИЛИЗИРА между кадрите -> pool кешът прескача излишните itemconfig
    try:
        key = (int((h % 1.0) * 48), int(s * 20), int(v * 20))
        c = _HSV_CACHE.get(key)
        if c:
            return c
        hq, sq, vq = key[0] / 48.0, key[1] / 20.0, key[2] / 20.0
        i = int(hq * 6)
        f = (hq * 6) - i
        p, q, t = vq * (1 - sq), vq * (1 - sq * f), vq * (1 - sq * (1 - f))
        rgb = [(vq, t, p), (q, vq, p), (p, vq, t), (p, q, vq), (t, p, vq), (vq, p, q)][
            i % 6
        ]
        c = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
        _HSV_CACHE[key] = c
        return c
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


def project_3d(x, y, z, cx, cy, size, dist=3.2):
    denom = max(0.2, dist - z)
    scale = dist / denom
    return cx + x * scale * size, cy - y * scale * size, scale


def depth_v(scale, lo=0.35, hi=1.0):
    """Яркост по дълбочина."""
    return max(lo, min(hi, 0.25 + scale * 0.55))


# =============================================================================
# STAR FIELD (стил 09)
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
# 1. HOPF FIBRATION — S3 като сноп от окръжности над S2
# =============================================================================


class HopfFibration:
    name = "Hopf Fibration"

    FIBERS = 22
    SEGS = 44

    def __init__(self):
        self.t = 0.0
        self.rot = 0.0

    def update(self, dt):
        self.t += dt * 0.25
        self.rot += dt * 0.3

    def get_lines(self, cx, cy, size):
        lines = []
        base_hue = self.t * 0.05
        for i in range(self.FIBERS):
            # базова точка на S2: широчината диша, дължината се върти
            eta = (0.5 + 0.36 * math.sin(self.t * 0.7 + i * 0.6)) * math.pi
            xi = TAU * i / self.FIBERS + self.t * 0.4
            ch, sh = math.cos(eta / 2), math.sin(eta / 2)
            hue = (base_hue + i / self.FIBERS) % 1.0
            prev = None
            for k in range(self.SEGS + 1):
                tt = TAU * k / self.SEGS
                a = ch * math.cos(tt)
                b = ch * math.sin(tt)
                c = sh * math.cos(tt + xi)
                d = sh * math.sin(tt + xi)
                denom = 1.0 - d
                if abs(denom) < 0.08:
                    prev = None
                    continue
                x, y, z = a / denom, b / denom, c / denom
                # мека компресия към екрана
                r = math.sqrt(x * x + y * y + z * z)
                f = 1.0 / (1.0 + 0.55 * r)
                x, y, z = x * f, y * f, z * f
                x, y, z = rotate_3d(x, y, z, ax=self.rot * 0.6, ay=self.rot)
                px, py, sc = project_3d(x, y, z, cx, cy, size * 1.35)
                if prev is not None:
                    lines.append(
                        (
                            prev[0],
                            prev[1],
                            px,
                            py,
                            hsv_to_hex(hue, 0.75, depth_v(sc)),
                            CONFIG["line_width"],
                        )
                    )
                prev = (px, py)
        return lines


# =============================================================================
# 2. THREE-BODY BALLET — хореографията "осмица"
# =============================================================================


class ThreeBodyBallet:
    name = "Three-Body Ballet"

    TRAIL = 260
    SUBSTEPS = 10
    DT = 0.004

    def __init__(self):
        self._reset()
        self.rot = 0.0
        self.hue0 = random.random()

    def _reset(self):
        # Chenciner-Montgomery фигура-8 начални условия
        p1 = [-0.97000436, 0.24308753]
        v3 = [-0.93240737, -0.86473146]
        self.pos = [p1[:], [-p1[0], -p1[1]], [0.0, 0.0]]
        self.vel = [[-v3[0] / 2, -v3[1] / 2], [-v3[0] / 2, -v3[1] / 2], v3[:]]
        self.trails = [[], [], []]
        self.age = 0.0
        # БЕЗ предзагряване (операторска преценка): формирането на осмицата
        # от нищото е самото зрелище

    def _advance(self):
        acc = self._acc()
        for _ in range(self.SUBSTEPS):
            for i in range(3):
                self.pos[i][0] += (
                    self.vel[i][0] * self.DT + 0.5 * acc[i][0] * self.DT**2
                )
                self.pos[i][1] += (
                    self.vel[i][1] * self.DT + 0.5 * acc[i][1] * self.DT**2
                )
            new_acc = self._acc()
            for i in range(3):
                self.vel[i][0] += 0.5 * (acc[i][0] + new_acc[i][0]) * self.DT
                self.vel[i][1] += 0.5 * (acc[i][1] + new_acc[i][1]) * self.DT
            acc = new_acc
        for i in range(3):
            self.trails[i].append((self.pos[i][0], self.pos[i][1]))
            if len(self.trails[i]) > self.TRAIL:
                self.trails[i].pop(0)

    def _acc(self):
        acc = [[0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                dx = self.pos[j][0] - self.pos[i][0]
                dy = self.pos[j][1] - self.pos[i][1]
                d2 = dx * dx + dy * dy + 1e-6
                inv = 1.0 / (d2 * math.sqrt(d2))
                acc[i][0] += dx * inv
                acc[i][1] += dy * inv
        return acc

    def update(self, dt):
        self.rot += dt * 0.22
        self.age += dt
        if self.age > 150:  # презареждане преди числен дрейф
            self._reset()
        self._advance()

    def get_lines(self, cx, cy, size):
        lines = []
        ca, sa = math.cos(self.rot), math.sin(self.rot)
        for i in range(3):
            hue = (self.hue0 + i / 3) % 1.0
            tr = self.trails[i]
            prev = None
            n = len(tr)
            for k, (x, y) in enumerate(tr):
                # леко 3D полюшване на равнината на танца
                X, Y, Z = rotate_3d(
                    x * 0.72,
                    y * 0.72,
                    0.0,
                    ax=0.35 * math.sin(self.rot * 0.5),
                    ay=self.rot * 0.15,
                )
                px = cx + (X * ca - Y * sa) * size * 0.62
                py = cy + (X * sa + Y * ca) * size * 0.62 - Z * size * 0.3
                if prev is not None:
                    fade = k / max(1, n)
                    lines.append(
                        (
                            prev[0],
                            prev[1],
                            px,
                            py,
                            hsv_to_hex(hue, 0.8, 0.25 + 0.7 * fade),
                            1 if fade < 0.7 else CONFIG["line_width"],
                        )
                    )
                prev = (px, py)
            # тялото: малък кръст (v1 вариантът, върнат по операторска преценка)
            if prev is not None:
                px, py = prev
                for ddx, ddy in ((-4, 0), (0, -4)):
                    lines.append(
                        (
                            px + ddx,
                            py + ddy,
                            px - ddx,
                            py - ddy,
                            hsv_to_hex(hue, 0.3, 1.0),
                            3,
                        )
                    )
        return lines


# =============================================================================
# 3. HYPERBOLIC FLOW — диск на Поанкаре под Мьобиус трансформация
# =============================================================================


class HyperbolicFlow:
    name = "Hyperbolic Flow"

    CHORDS = 42
    SEGS = 26

    def __init__(self):
        self.t = 0.0
        random.seed(7)
        self.pairs = []
        for k in range(self.CHORDS):
            a = TAU * k / self.CHORDS
            b = a + random.uniform(0.5, 2.6)
            self.pairs.append((a, b))

    def update(self, dt):
        self.t += dt * 0.3

    def _mobius(self, zx, zy):
        # T(z) = e^{i phi} (z + a) / (1 + conj(a) z),  a = малка орбитираща точка
        ar = 0.34 * math.cos(self.t * 0.6)
        ai = 0.34 * math.sin(self.t * 0.6)
        nx, ny = zx + ar, zy + ai
        dx, dy = 1.0 + ar * zx + ai * zy, ar * zy - ai * zx
        d2 = dx * dx + dy * dy
        if d2 < 1e-9:
            return 0.0, 0.0
        rx, ry = (nx * dx + ny * dy) / d2, (ny * dx - nx * dy) / d2
        phi = self.t * 0.25
        cp, sp = math.cos(phi), math.sin(phi)
        return rx * cp - ry * sp, rx * sp + ry * cp

    def get_lines(self, cx, cy, size):
        lines = []
        R = size * 0.92
        # граничната окръжност
        prev = None
        for k in range(73):
            a = TAU * k / 72
            px, py = cx + R * math.cos(a), cy + R * math.sin(a)
            if prev:
                lines.append(
                    (prev[0], prev[1], px, py, hsv_to_hex(self.t * 0.03, 0.2, 0.45), 1)
                )
            prev = (px, py)
        # геодезичните
        for idx, (a, b) in enumerate(self.pairs):
            ux, uy = math.cos(a), math.sin(a)
            vx, vy = math.cos(b), math.sin(b)
            dot = ux * vx + uy * vy
            hue = (idx / self.CHORDS + self.t * 0.04) % 1.0
            prev = None
            if dot > 0.999:
                continue
            gcx, gcy = (ux + vx) / (1 + dot), (uy + vy) / (1 + dot)
            r = math.sqrt(max(1e-9, gcx * gcx + gcy * gcy - 1.0))
            a1 = math.atan2(uy - gcy, ux - gcx)
            a2 = math.atan2(vy - gcy, vx - gcx)
            while a2 - a1 > math.pi:
                a2 -= TAU
            while a1 - a2 > math.pi:
                a2 += TAU
            for k in range(self.SEGS + 1):
                th = a1 + (a2 - a1) * k / self.SEGS
                zx, zy = gcx + r * math.cos(th), gcy + r * math.sin(th)
                zx, zy = self._mobius(zx, zy)
                rr = zx * zx + zy * zy
                if rr > 0.985:
                    prev = None
                    continue
                px, py = cx + zx * R, cy + zy * R
                if prev is not None:
                    # по-ярко към центъра (хиперболичната метрика "гори" в средата)
                    v = 0.5 + 0.5 * (1.0 - rr)
                    lines.append((prev[0], prev[1], px, py, hsv_to_hex(hue, 0.7, v), 2))
                prev = (px, py)
        return lines


# =============================================================================
# 4. CLIFFORD ATTRACTOR — морфиращ странен атрактор
# =============================================================================


class CliffordAttractor:
    name = "Clifford Attractor"

    POINTS = 900  # по-малко, но ПО-ЕДРИ и ярки точки -> и по-бърз render

    def __init__(self):
        self.t = 0.0
        self.x, self.y = 0.1, 0.0

    def update(self, dt):
        self.t += dt * 0.38  # видимо по-жива морфология

    def get_lines(self, cx, cy, size):
        # параметрите дишат около известна добра точка
        a = -1.4 + 0.30 * math.sin(self.t * 0.9)
        b = 1.6 + 0.30 * math.sin(self.t * 0.7 + 1.0)
        c = 1.0 + 0.25 * math.sin(self.t * 0.5 + 2.0)
        d = 0.7 + 0.25 * math.sin(self.t * 1.1 + 4.0)
        lines = []
        x, y = self.x, self.y
        s = size * 0.44
        hue0 = self.t * 0.10
        for i in range(self.POINTS):
            nx = math.sin(a * y) + c * math.cos(a * x)
            ny = math.sin(b * x) + d * math.cos(b * y)
            x, y = nx, ny
            if i < 12:  # прескачаме прехода
                continue
            px, py = cx + x * s, cy + y * s
            hue = (hue0 + i / self.POINTS * 0.35) % 1.0
            v = 0.8 + 0.2 * (i / self.POINTS)
            lines.append((px - 0.5, py, px + 0.5, py, hsv_to_hex(hue, 0.9, v), 3))
        self.x, self.y = x, y
        return lines


# =============================================================================
# 5. CHAOS BUTTERFLIES — рояк двойни махала (чувствителност 1e-5)
# =============================================================================


class ChaosButterflies:
    name = "Chaos Butterflies"

    N = 36
    TRAIL = 26
    G = 9.81
    SUBSTEPS = 3
    DT = 0.011

    def __init__(self):
        self._reset()

    def _reset(self):
        th = random.uniform(1.9, 2.6)
        self.state = [[th + i * 1e-5, 0.0, th * 0.6, 0.0] for i in range(self.N)]
        self.trails = [[] for _ in range(self.N)]
        self.age = 0.0
        self.hue0 = random.random()

    def _deriv(self, s):
        t1, w1, t2, w2 = s
        al = t1 - t2
        den = 3.0 - math.cos(2 * al)
        a1 = (
            -3 * self.G * math.sin(t1)
            - self.G * math.sin(t1 - 2 * t2)
            - 2 * math.sin(al) * (w2 * w2 + w1 * w1 * math.cos(al))
        ) / den
        a2 = (
            2
            * math.sin(al)
            * (2 * w1 * w1 + 2 * self.G * math.cos(t1) + w2 * w2 * math.cos(al))
        ) / den
        return w1, a1, w2, a2

    def update(self, dt):
        self.age += dt
        if self.age > 95:
            self._reset()
        for idx in range(self.N):
            s = self.state[idx]
            for _ in range(self.SUBSTEPS):
                # RK4
                k1 = self._deriv(s)
                s2 = [s[i] + k1[i] * self.DT / 2 for i in range(4)]
                k2 = self._deriv(s2)
                s3 = [s[i] + k2[i] * self.DT / 2 for i in range(4)]
                k3 = self._deriv(s3)
                s4 = [s[i] + k3[i] * self.DT for i in range(4)]
                k4 = self._deriv(s4)
                for i in range(4):
                    s[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * self.DT / 6
            t1, _, t2, _ = s
            tip = (math.sin(t1) + math.sin(t2), math.cos(t1) + math.cos(t2))
            tr = self.trails[idx]
            tr.append(tip)
            if len(tr) > self.TRAIL:
                tr.pop(0)

    def get_lines(self, cx, cy, size):
        lines = []
        L = size * 0.30
        oy = cy - size * 0.22
        for idx in range(self.N):
            t1, _, t2, _ = self.state[idx]
            hue = (self.hue0 + idx / self.N) % 1.0
            x1, y1 = cx + L * math.sin(t1), oy + L * math.cos(t1)
            x2, y2 = x1 + L * math.sin(t2), y1 + L * math.cos(t2)
            lines.append((cx, oy, x1, y1, hsv_to_hex(hue, 0.25, 0.5), 1))
            lines.append(
                (x1, y1, x2, y2, hsv_to_hex(hue, 0.6, 0.85), CONFIG["line_width"])
            )
            prev = None
            tr = self.trails[idx]
            n = len(tr)
            for k, (tx, ty) in enumerate(tr):
                px, py = cx + tx * L, oy + ty * L
                if prev is not None:
                    fade = k / max(1, n)
                    lines.append(
                        (
                            prev[0],
                            prev[1],
                            px,
                            py,
                            hsv_to_hex(hue, 0.9, 0.15 + 0.65 * fade),
                            1,
                        )
                    )
                prev = (px, py)
        return lines


# =============================================================================
# 6. KLEIN BOTTLE — bagel immersion, въртяща се
# =============================================================================


class KleinBottle:
    name = "Klein Bottle"

    NU = 30
    NV = 16

    def __init__(self):
        self.rot = 0.0
        self.t = 0.0

    def _point(self, u, v):
        # "bagel" вложение на бутилката на Клайн
        cu, su = math.cos(u), math.sin(u)
        r = 4 * (1 - cu / 2)
        if u < math.pi:
            x = 6 * cu * (1 + su) + r * math.cos(v) * cu
            y = 16 * su + r * math.cos(v) * su
        else:
            x = 6 * cu * (1 + su) + r * math.cos(v + math.pi)
            y = 16 * su
        z = r * math.sin(v)
        return x / 21.0, y / 21.0, z / 21.0

    def update(self, dt):
        self.rot += dt * 0.35
        self.t += dt

    def get_lines(self, cx, cy, size):
        lines = []
        pts = {}
        breath = 1.0 + 0.05 * math.sin(self.t * 0.8)
        for i in range(self.NU + 1):
            u = TAU * i / self.NU
            for j in range(self.NV):
                x, y, z = self._point(u, TAU * j / self.NV)
                x, y, z = rotate_3d(
                    x * breath,
                    y * breath,
                    z * breath,
                    ax=self.rot * 0.7,
                    ay=self.rot,
                    az=self.rot * 0.23,
                )
                pts[(i, j)] = project_3d(x, y, z, cx, cy, size * 0.9)
        hue0 = self.t * 0.05
        for i in range(self.NU + 1):
            for j in range(self.NV):
                p = pts[(i, j)]
                hue = (hue0 + j / self.NV) % 1.0
                if i < self.NU:
                    q = pts[(i + 1, j)]
                    lines.append(
                        (p[0], p[1], q[0], q[1], hsv_to_hex(hue, 0.7, depth_v(p[2])), 1)
                    )
                q = pts[(i, (j + 1) % self.NV)]
                lines.append(
                    (
                        p[0],
                        p[1],
                        q[0],
                        q[1],
                        hsv_to_hex((hue + 0.04) % 1.0, 0.7, depth_v(p[2])),
                        1,
                    )
                )
        return lines


# =============================================================================
# 7. TORUS KNOT MORPH — (p,q) възли, преливащи един в друг
# =============================================================================


class TorusKnotMorph:
    name = "Torus Knot Morph"

    SEQ = [(2, 3), (3, 4), (2, 5), (3, 5), (4, 3), (2, 7)]
    SEGS = 420
    HOLD = 9.0
    MORPH = 3.0

    def __init__(self):
        self.t = 0.0
        self.rot = 0.0

    def update(self, dt):
        self.t += dt
        self.rot += dt * 0.4

    def _pq(self):
        cycle = self.HOLD + self.MORPH
        total = self.t % (cycle * len(self.SEQ))
        idx = int(total // cycle)
        local = total - idx * cycle
        p1, q1 = self.SEQ[idx]
        p2, q2 = self.SEQ[(idx + 1) % len(self.SEQ)]
        if local < self.HOLD:
            return float(p1), float(q1)
        s = (local - self.HOLD) / self.MORPH
        s = s * s * (3 - 2 * s)  # smoothstep
        return p1 + (p2 - p1) * s, q1 + (q2 - q1) * s

    def get_lines(self, cx, cy, size):
        p, q = self._pq()
        lines = []
        prev = None
        hue0 = self.t * 0.04
        for k in range(self.SEGS + 1):
            th = TAU * k / self.SEGS
            r = math.cos(q * th) + 2.0
            x = r * math.cos(p * th) / 3.2
            y = r * math.sin(p * th) / 3.2
            z = -math.sin(q * th) / 3.2
            x, y, z = rotate_3d(x, y, z, ax=self.rot * 0.6, ay=self.rot)
            px, py, sc = project_3d(x, y, z, cx, cy, size * 1.15)
            if prev is not None:
                hue = (hue0 + k / self.SEGS) % 1.0
                lines.append(
                    (
                        prev[0],
                        prev[1],
                        px,
                        py,
                        hsv_to_hex(hue, 0.8, depth_v(sc)),
                        CONFIG["line_width"],
                    )
                )
            prev = (px, py)
        return lines


# =============================================================================
# 8. MURMURATION WARS — три воюващи ята (камък-ножица-хартия)
# =============================================================================


class MurmurationWars:
    name = "Murmuration Wars"

    N = 135  # общо, ~45 на ято
    FLOCKS = 3
    HUES = (0.02, 0.36, 0.58)  # червено / зелено / синьо-виолетово
    SPEED = 3.6
    NEIGH = 6400  # d^2 праг за свои
    HUNT2 = 48400  # 220px обхват на лова
    FLEE2 = 25600  # 160px паника
    CATCH2 = 196  # 14px = улов -> конверсия
    MIN_FLOCK = 6  # под този брой ятото не губи членове

    def __init__(self):
        self.t = 0.0
        self.w, self.h = 1200, 800
        self.boids = []
        for f in range(self.FLOCKS):
            hx = self.w * (0.25 + 0.25 * f)
            hy = self.h * (0.3 + 0.2 * (f % 2))
            for _ in range(self.N // self.FLOCKS):
                self.boids.append(
                    [
                        hx + random.uniform(-90, 90),
                        hy + random.uniform(-90, 90),
                        random.uniform(-2, 2),
                        random.uniform(-2, 2),
                        f,
                    ]
                )
        self.bursts = []  # (x, y, age, hue) — пръстени при улов

    def _counts(self):
        c = [0] * self.FLOCKS
        for b in self.boids:
            c[b[4]] += 1
        return c

    def update(self, dt):
        self.t += dt
        counts = self._counts()
        # три блуждаещи цели — ятата се разминават и сблъскват
        anchors = []
        for f in range(self.FLOCKS):
            ph = self.t * 0.16 + f * TAU / 3
            anchors.append(
                (
                    self.w / 2 + math.cos(ph) * self.w * 0.3,
                    self.h / 2 + math.sin(ph * 1.3) * self.h * 0.28,
                )
            )
        conversions = []
        for i, b in enumerate(self.boids):
            f = b[4]
            prey, pred = (f + 1) % 3, (f + 2) % 3
            sx = sy = alx = aly = cxx = cyy = 0.0
            cnt = 0
            hunt = None
            hunt_d2 = self.HUNT2
            flee_x = flee_y = 0.0
            for j, o in enumerate(self.boids):
                if i == j:
                    continue
                dx, dy = o[0] - b[0], o[1] - b[1]
                d2 = dx * dx + dy * dy
                of = o[4]
                if of == f:
                    if d2 < self.NEIGH:
                        cnt += 1
                        alx += o[2]
                        aly += o[3]
                        cxx += o[0]
                        cyy += o[1]
                        if d2 < 620:
                            k = 1.0 / max(8.0, d2)
                            sx -= dx * k
                            sy -= dy * k
                elif of == prey:
                    if d2 < hunt_d2:
                        hunt_d2 = d2
                        hunt = (dx, dy, d2, j)
                elif of == pred:
                    if d2 < self.FLEE2:
                        k = 1.0 / max(20.0, d2)
                        flee_x -= dx * k
                        flee_y -= dy * k
            if cnt:
                b[2] += (alx / cnt - b[2]) * 0.045 + (cxx / cnt - b[0]) * 0.0022
                b[3] += (aly / cnt - b[3]) * 0.045 + (cyy / cnt - b[1]) * 0.0022
            b[2] += sx * 14 + flee_x * 34
            b[3] += sy * 14 + flee_y * 34
            if hunt is not None:
                dx, dy, d2, j = hunt
                d = math.sqrt(d2) or 1.0
                b[2] += dx / d * 0.10
                b[3] += dy / d * 0.10
                if d2 < self.CATCH2 and counts[prey] > self.MIN_FLOCK:
                    conversions.append((j, f))
                    counts[prey] -= 1
                    counts[f] += 1
            ax, ay = anchors[f]
            b[2] += (ax - b[0]) * 0.0005
            b[3] += (ay - b[1]) * 0.0005
            sp = math.sqrt(b[2] ** 2 + b[3] ** 2) or 1.0
            lim = self.SPEED * (1.1 + 0.3 * math.sin(self.t * 1.3 + i))
            if sp > lim:
                b[2], b[3] = b[2] / sp * lim, b[3] / sp * lim
            b[0] = (b[0] + b[2] * 60 * dt) % self.w
            b[1] = (b[1] + b[3] * 60 * dt) % self.h
        # улов -> жертвата сменя знамето + пръстен
        for j, new_f in conversions:
            victim = self.boids[j]
            if victim[4] != new_f:
                victim[4] = new_f
                self.bursts.append([victim[0], victim[1], 0.0, self.HUES[new_f]])
        self.bursts = [[x, y, age + dt, h] for x, y, age, h in self.bursts if age < 0.6]

    def get_lines(self, cx, cy, size):
        lines = []
        sx_f = (cx * 2) / self.w if self.w else 1.0
        sy_f = (cy * 2) / self.h if self.h else 1.0
        # плътни големи точки (capstyle=round) + ореол в цвета на ятото
        for i, b in enumerate(self.boids):
            hue = (self.HUES[b[4]] + 0.03 * math.sin(self.t * 2 + i)) % 1.0
            px, py = b[0] * sx_f, b[1] * sy_f
            lines.append((px - 0.5, py, px + 0.5, py, hsv_to_hex(hue, 0.5, 0.4), 13))
            lines.append((px - 0.5, py, px + 0.5, py, hsv_to_hex(hue, 0.75, 1.0), 8))
        # пръстени при конверсия
        for x, y, age, h in self.bursts:
            r = 6 + age * 90
            v = max(0.0, 1.0 - age / 0.6)
            px, py = x * sx_f, y * sy_f
            prev = None
            for k in range(13):
                a = TAU * k / 12
                qx, qy = px + r * math.cos(a), py + r * math.sin(a)
                if prev:
                    lines.append((prev[0], prev[1], qx, qy, hsv_to_hex(h, 0.6, v), 2))
                prev = (qx, qy)
        return lines


# =============================================================================
# MAIN APP — line pool, часовник, авто-превключване (стил 09)
# =============================================================================

# Пълният цикъл от 8, в реда от описанието (ChaosButterflies върнат 14.07 следобед)
VISUALIZATIONS = [
    HopfFibration,
    ThreeBodyBallet,
    HyperbolicFlow,
    CliffordAttractor,
    ChaosButterflies,
    KleinBottle,
    TorusKnotMorph,
    MurmurationWars,
]


try:
    import psutil  # телеметрия на машината (наличен в env-а; graceful без него)
except Exception:
    psutil = None


class ScreensaverApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Telemetry")  # неутрално заглавие за любопитни очи
        self.root.configure(bg=CONFIG["bg_color"])
        # ПРОЗОРЕЦ, не fullscreen: покрива Outlook на втория екран, мести се свободно
        self.root.geometry("1280x800+120+60")
        self.root.minsize(480, 320)
        self.canvas = tk.Canvas(self.root, bg=CONFIG["bg_color"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.width, self.height = 1280, 800
        self.stars = StarField(self.width, self.height)

        # pools + кеш на състоянието (itemconfig само при реална промяна)
        self.line_pool = [
            self.canvas.create_line(0, 0, 0, 0, state="hidden", capstyle="round")
            for _ in range(CONFIG["max_lines"])
        ]
        self.pool_style = [None] * CONFIG["max_lines"]  # (color, width) кеш
        self.pool_shown = [False] * CONFIG["max_lines"]  # state кеш
        self.star_pool = [
            self.canvas.create_oval(0, 0, 0, 0, outline="")
            for _ in range(CONFIG["num_stars"])
        ]
        self.clock_id = self.canvas.create_text(
            self.width // 2,
            58,
            fill="#ffffff",
            font=("Consolas", CONFIG["clock_font_size"], "bold"),
            text="",
        )
        self.date_id = self.canvas.create_text(
            self.width // 2, 104, fill="#8a94aa", font=("Consolas", 14, "bold"), text=""
        )
        self.name_id = self.canvas.create_text(
            self.width - 16,
            self.height - 14,
            anchor="e",
            fill="#3a4458",
            font=("Consolas", 11),
            text="",
        )
        self.stats_id = self.canvas.create_text(
            14,
            14,
            anchor="nw",
            fill="#6a768e",
            font=("Consolas", 12, "bold"),
            text="",
            justify="left",
        )
        self._stats_next = 0.0

        self.viz_index = 0  # старт от №1, предвидим цикъл
        self.viz = VISUALIZATIONS[0]()
        self.last_time = time.time()

        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self._switch())
        self.root.bind("<Right>", lambda e: self._switch())
        self.root.bind("<Left>", lambda e: self._switch(-1))
        # клик върху платното = следваща (работи и когато прозорецът не е имал фокус)
        self.canvas.bind("<Button-1>", lambda e: self._switch())
        self.canvas.bind("<Configure>", self._on_resize)
        self.root.after(CONFIG["switch_interval"], self._auto_switch)

    def _on_resize(self, event):
        self.width, self.height = max(1, event.width), max(1, event.height)
        self.stars.resize(self.width, self.height)
        self.canvas.coords(self.clock_id, self.width // 2, 58)
        self.canvas.coords(self.date_id, self.width // 2, 104)
        self.canvas.coords(self.name_id, self.width - 16, self.height - 14)

    def _stats_text(self):
        if psutil is None:
            return ""
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("C:\\")
            return (
                f"CPU {cpu:4.0f}%   RAM {mem.percent:4.0f}%   "
                f"C: {disk.free / 2**30:.0f}G free"
            )
        except Exception:
            return ""

    def _switch(self, step=1):
        self.viz_index = (self.viz_index + step) % len(VISUALIZATIONS)
        self.viz = VISUALIZATIONS[self.viz_index]()

    def _auto_switch(self):
        self._switch()
        self.root.after(CONFIG["switch_interval"], self._auto_switch)

    def _frame(self):
        now = time.time()
        dt = min(0.1, now - self.last_time)
        self.last_time = now

        self.stars.update()
        self.viz.update(dt)

        # звезди
        sz = CONFIG["star_size"]
        for item, (x, y, color) in zip(self.star_pool, self.stars.get_stars()):
            self.canvas.coords(item, x, y, x + sz, y + sz)
            self.canvas.itemconfig(item, fill=color)

        # линии (кеширан pool: itemconfig само при промяна на стил/видимост)
        cx, cy = self.width / 2, self.height / 2 + 20
        size = min(self.width, self.height) * 0.36
        lines = self.viz.get_lines(cx, cy, size)
        pool, style, shown = self.line_pool, self.pool_style, self.pool_shown
        canvas = self.canvas
        n = min(len(lines), len(pool))
        for i in range(n):
            x1, y1, x2, y2, color, w = lines[i]
            canvas.coords(pool[i], x1, y1, x2, y2)
            st = (color, w)
            if style[i] != st:
                canvas.itemconfig(pool[i], fill=color, width=w)
                style[i] = st
            if not shown[i]:
                canvas.itemconfig(pool[i], state="normal")
                shown[i] = True
        for i in range(n, len(pool)):
            if shown[i]:
                canvas.itemconfig(pool[i], state="hidden")
                shown[i] = False

        # часовник + телеметрия (1 Hz)
        now_dt = datetime.now()
        self.canvas.itemconfig(self.clock_id, text=now_dt.strftime("%H:%M:%S"))
        self.canvas.itemconfig(self.date_id, text=now_dt.strftime("%A, %d %B %Y"))
        self.canvas.itemconfig(
            self.name_id,
            text=f"{self.viz_index + 1}/{len(VISUALIZATIONS)} · {self.viz.name}",
        )
        if now >= self._stats_next:
            self._stats_next = now + 1.0
            self.canvas.itemconfig(self.stats_id, text=self._stats_text())

        self.root.after(1000 // CONFIG["fps"], self._frame)

    def run(self):
        self._frame()
        self.root.mainloop()


# =============================================================================
# SELFTEST — без екран: 60 кадъра на всяка визуализация
# =============================================================================


def selftest():
    print(f"Selftest: 60 кадъра x {len(VISUALIZATIONS)} визуализации (без екран)")
    ok = True
    for cls in VISUALIZATIONS:
        viz = cls()
        t0 = time.time()
        max_lines = 0
        try:
            for _ in range(60):
                viz.update(1 / 30)
                lines = viz.get_lines(960, 540, 380)
                max_lines = max(max_lines, len(lines))
                for ln in lines:
                    assert len(ln) == 6
                    for v in ln[:4]:
                        assert math.isfinite(v), f"{cls.name}: non-finite coord"
        except Exception as e:
            ok = False
            print(f"  FAIL {cls.name}: {type(e).__name__}: {e}")
            continue
        ms = (time.time() - t0) / 60 * 1000
        status = "OK " if max_lines <= CONFIG["max_lines"] else "OVER-POOL"
        print(f"  {status} {cls.name:22} max_lines={max_lines:5}  {ms:6.2f} ms/frame")
        if max_lines > CONFIG["max_lines"]:
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ScreensaverApp().run()
