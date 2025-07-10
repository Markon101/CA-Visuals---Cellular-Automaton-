# --- ANNOTATED GENESIS ENGINE v5.2 ---
# --- API & Sampling Correction Update ---
#
# This version fixes the crash from v5.1 by implementing a custom
# sampling function, as `field.sample()` is not a valid method.
#
# Key Improvements from v5.1:
# 1. [FIX] AttributeError Crash: Removed the call to the non-existent
#    `.sample()` method.
# 2. [NEW] Custom Bilinear Sampling Function: Created a new Taichi function
#    `sample_wrap` that performs bilinear interpolation with wrapping boundary
#    conditions manually. This correctly implements the intended advection
#    logic and resolves the crash.

import pygame
import numpy as np
import librosa
import taichi as ti
import taichi.math as tm

# --- TAICHI INITIALIZATION ---
ti.init(arch=ti.gpu)

# --- CONFIGURATION ---
SONG_FILE = 'Singularity Anthem.wav'
RESOLUTION_SCALE = 1.0
FPS = 60

# --- AESTHETIC & REACTIVITY TUNING ---
TURBULENCE_STRENGTH = 8.0
TRANSIENT_BURST_STRENGTH = 250.0
MAX_FLOW_SPEED = 25.0
FRACTAL_SCALE = 0.004
SIMULATION_TIMESTEP = 0.08
INNER_SIMULATION_STEPS = 12

# --- DERIVED CONSTANTS ---
WIDTH = int(1024 * RESOLUTION_SCALE)
HEIGHT = int(1024 * RESOLUTION_SCALE)

# --- TAICHI DATA STRUCTURES ---
U = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
V = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
U_new = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
V_new = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
feed_map = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
kill_map = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
flow_field = ti.Vector.field(2, dtype=ti.f32, shape=(WIDTH, HEIGHT))
pixel_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
palette_base_color = ti.Vector.field(3, dtype=ti.f32, shape=())
palette_u_color = ti.Vector.field(3, dtype=ti.f32, shape=())
palette_v_color = ti.Vector.field(3, dtype=ti.f32, shape=())


# --- TAICHI KERNELS & FUNCS ---

# [NEW] Custom sampling function to replace the incorrect .sample() method.
# This function performs bilinear interpolation with wrapping boundary conditions.
@ti.func
def sample_wrap(field, u, v):
    iu, iv = int(u), int(v)
    fu, fv = u - iu, v - iv
    
    # Wrap coordinates using modulo
    iu0 = iu % WIDTH
    iu1 = (iu + 1) % WIDTH
    iv0 = iv % HEIGHT
    iv1 = (iv + 1) % HEIGHT

    # Bilinear interpolation
    c00 = field[iu0, iv0]
    c10 = field[iu1, iv0]
    c01 = field[iu0, iv1]
    c11 = field[iu1, iv1]
    
    return tm.mix(tm.mix(c00, c10, fu), tm.mix(c01, c11, fu), fv)


@ti.func
def smoothstep(edge0, edge1, x):
    t = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def perlin_noise(p, time):
    pi = tm.floor(p)
    pf = p - pi
    w = pf * pf * (3.0 - 2.0 * pf)
    g00 = tm.normalize(tm.vec2(tm.sin(pi.x * 7.1 + time), tm.cos(pi.y * 6.3 - time)))
    g10 = tm.normalize(tm.vec2(tm.sin((pi.x + 1.0) * 7.1 + time), tm.cos(pi.y * 6.3 - time)))
    g01 = tm.normalize(tm.vec2(tm.sin(pi.x * 7.1 + time), tm.cos((pi.y + 1.0) * 6.3 - time)))
    g11 = tm.normalize(tm.vec2(tm.sin((pi.x + 1.0) * 7.1 + time), tm.cos((pi.y + 1.0) * 6.3 - time)))
    n00, n10 = g00.dot(pf), g10.dot(pf - tm.vec2(1.0, 0.0))
    n01, n11 = g01.dot(pf - tm.vec2(0.0, 1.0)), g11.dot(pf - tm.vec2(1.0, 1.0))
    return tm.mix(tm.mix(n00, n10, w.x), tm.mix(n01, n11, w.x), w.y)

@ti.func
def fractal_noise(p, time, octaves: ti.template()):
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    for _ in ti.static(range(octaves)):
        value += amplitude * perlin_noise(p * frequency, time)
        amplitude *= 0.5
        frequency *= 2.0
    return value

@ti.kernel
def update_flow_field_kernel(time: ti.f32, transient_energy: ti.f32):
    for i, j in flow_field:
        angle = perlin_noise(tm.vec2(i,j) * FRACTAL_SCALE * 0.5, time * 0.1) * tm.pi * 2.0
        base_flow = tm.vec2(tm.cos(angle), tm.sin(angle)) * TURBULENCE_STRENGTH
        burst_angle = perlin_noise(tm.vec2(i,j) * FRACTAL_SCALE * 2.0, time) * tm.pi * 2.0
        burst_flow = tm.vec2(tm.cos(burst_angle), tm.sin(burst_angle))
        burst_strength = transient_energy * transient_energy * TRANSIENT_BURST_STRENGTH
        flow_field[i, j] = base_flow + burst_flow * burst_strength

@ti.kernel
def clamp_flow_field_kernel():
    for i, j in flow_field:
        mag_sq = flow_field[i, j].norm_sqr()
        if mag_sq > MAX_FLOW_SPEED * MAX_FLOW_SPEED:
            flow_field[i, j] = flow_field[i, j].normalized() * MAX_FLOW_SPEED

@ti.kernel
def update_physics_map_kernel(time: ti.f32, feed_base: ti.f32, feed_mod: ti.f32, kill_base: ti.f32, kill_mod: ti.f32):
    for i, j in feed_map:
        noise_val = fractal_noise(tm.vec2(i, j) * FRACTAL_SCALE, time * 0.2, 4)
        feed_map[i, j] = feed_base + noise_val * feed_mod
        kill_map[i, j] = kill_base + noise_val * kill_mod

@ti.kernel
def calculate_reaction_kernel(dt: ti.f32):
    Du, Dv = 1.0, 0.5
    for i, j in U:
        ip1, im1 = (i + 1) % WIDTH, (i - 1 + WIDTH) % WIDTH
        jp1, jm1 = (j + 1) % HEIGHT, (j - 1 + HEIGHT) % HEIGHT
        
        center_U, center_V = U[i, j], V[i, j]
        laplacian_U = (U[ip1, j] + U[im1, j] + U[i, jp1] + U[i, jm1] + \
                       U[ip1, jp1] + U[im1, jp1] + U[ip1, jm1] + U[im1, jm1] - 8 * center_U)
        laplacian_V = (V[ip1, j] + V[im1, j] + V[i, jp1] + V[i, jm1] + \
                       V[ip1, jp1] + V[im1, jp1] + V[ip1, jm1] + V[im1, jm1] - 8 * center_V)

        warp_pos = tm.vec2(i, j) - flow_field[i, j] * dt
        
        # [FIX] Use the new custom sampling function.
        advected_U = sample_wrap(U, warp_pos.x, warp_pos.y)
        advected_V = sample_wrap(V, warp_pos.x, warp_pos.y)

        uvv = advected_U * advected_V * advected_V
        feed, kill = feed_map[i, j], kill_map[i, j]
        
        U_new[i, j] = advected_U + (Du * laplacian_U - uvv + feed * (1.0 - advected_U)) * dt
        V_new[i, j] = advected_V + (Dv * laplacian_V + uvv - (feed + kill) * advected_V) * dt
        
        U_new[i, j] = tm.clamp(U_new[i, j], 0.0, 1.0)
        V_new[i, j] = tm.clamp(V_new[i, j], 0.0, 1.0)

@ti.kernel
def copy_state_kernel():
    for i, j in U: U[i, j], V[i, j] = U_new[i, j], V_new[i, j]

@ti.kernel
def render_kernel():
    for i, j in pixel_buffer:
        u, v = U[i, j], V[i, j]
        color = palette_base_color[None]
        color = tm.mix(color, palette_u_color[None], smoothstep(0.1, 0.6, u))
        color = tm.mix(color, palette_v_color[None], smoothstep(0.1, 0.5, v))
        glow = tm.vec3(1.0, 0.9, 0.7) * tm.pow(v, 2.0)
        pixel_buffer[i, j] = tm.pow(color + glow, 1.1)

@ti.kernel
def init_simulation():
    U.fill(1.0); V.fill(0.0); flow_field.fill(0)
    size = 20
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    for i, j in ti.ndrange((center_x - size, center_x + size), (center_y - size, center_y + size)):
        if (i-center_x)**2 + (j-center_y)**2 < size**2:
            V[i, j] = 0.8

@ti.kernel
def splash(x: int, y: int):
    size = 40
    for i, j in ti.ndrange((-size, size), (-size, size)):
        if (i*i + j*j) < size**2:
            px, py = (x + i + WIDTH) % WIDTH, (y + j + HEIGHT) % HEIGHT
            V[px, py] = 0.95

# --- PYTHON-SIDE CLASSES ---
class MusicAnalyzer:
    def __init__(self, song_path):
        try:
            print(f"Performing deep analysis of '{song_path}'...")
            y, self.sr = librosa.load(song_path, sr=44100)
            self.duration = librosa.get_duration(y=y, sr=self.sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            self.transient_energy = self._normalize(onset_env)
            chromagram = librosa.feature.chroma_stft(y=y, sr=self.sr)
            self.chroma_features = self._normalize_chroma(chromagram.T)
            self.times = librosa.times_like(self.chroma_features, sr=self.sr)
            print("Analysis complete.")
        except Exception as e: print(f"ERROR: Could not load song '{song_path}'. {e}"); exit()
    def _normalize(self, data): return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
    def _normalize_chroma(self, chroma): return chroma / (np.max(chroma, axis=1, keepdims=True) + 1e-6)
    def get_data_at_time(self, t):
        t %= self.duration
        frame_idx = np.searchsorted(self.times, t, side='right') -1
        return {'transients': self.transient_energy[frame_idx], 'chroma': self.chroma_features[frame_idx]}

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.music = MusicAnalyzer(SONG_FILE)
        
        self.palettes = {
            "inferno": np.array([[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [1.0, 0.9, 0.5]], dtype=np.float32),
            "twilight": np.array([[0.1, 0.0, 0.2], [0.8, 0.2, 0.4], [0.2, 0.7, 1.0]], dtype=np.float32),
            "oceanic": np.array([[0.0, 0.1, 0.2], [0.1, 0.5, 0.6], [0.8, 1.0, 0.9]], dtype=np.float32)
        }
        self.current_palette = self.palettes["inferno"].copy()
        self.target_palette_name = "inferno"

        init_simulation()

    def run(self):
        try: pygame.mixer.music.load(SONG_FILE); pygame.mixer.music.play(-1)
        except pygame.error: print(f"WARNING: Pygame could not play music file '{SONG_FILE}'.")
        
        running, time = True, 0.0
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            if dt > 0.1: dt = 0.1
            time += dt
            
            music_time = pygame.mixer.music.get_pos() / 1000.0
            music_data = self.music.get_data_at_time(music_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEBUTTONDOWN: splash(event.pos[0], event.pos[1])
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: init_simulation()
            
            self.update(music_data, time, dt)
            self.draw()
        pygame.quit()

    def update(self, music_data, time, dt):
        transients, chroma = music_data['transients'], music_data['chroma']

        update_flow_field_kernel(time, transients)
        clamp_flow_field_kernel()
        
        bass_notes = chroma[0] + chroma[1]
        mid_notes = chroma[5] + chroma[6]
        high_notes = chroma[10] + chroma[11]
        
        feed_base = 0.015 + bass_notes * 0.04
        feed_mod = 0.03 + mid_notes * 0.03
        kill_base = 0.045 + high_notes * 0.01
        kill_mod = 0.015 + (1.0 - bass_notes) * 0.01
        
        update_physics_map_kernel(time, feed_base, feed_mod, kill_base, kill_mod)
        
        for _ in range(INNER_SIMULATION_STEPS):
             calculate_reaction_kernel(SIMULATION_TIMESTEP)
             copy_state_kernel()

        total_chroma_energy = chroma.sum() / 12.0
        if total_chroma_energy < 0.2: self.target_palette_name = "twilight"
        elif total_chroma_energy > 0.6 and transients > 0.5: self.target_palette_name = "inferno"
        elif high_notes > 0.7: self.target_palette_name = "oceanic"
        
        target_colors = self.palettes[self.target_palette_name]
        lerp_rate = min(2.0 * dt, 1.0)
        self.current_palette += (target_colors - self.current_palette) * lerp_rate

        palette_base_color.from_numpy(self.current_palette[0])
        palette_u_color.from_numpy(self.current_palette[1])
        palette_v_color.from_numpy(self.current_palette[2])

    def draw(self):
        render_kernel()
        pixels_np = (np.clip(pixel_buffer.to_numpy(), 0, 1) * 255).astype(np.uint8)
        pygame.surfarray.blit_array(self.screen, np.transpose(pixels_np, (1, 0, 2)))
        pygame.display.flip()
        pygame.display.set_caption(f"Genesis Engine v5.2 | FPS: {self.clock.get_fps():.1f}")

if __name__ == '__main__':
    Visualizer().run()
