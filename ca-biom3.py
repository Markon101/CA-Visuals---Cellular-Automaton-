# --- README ---
# The Synesthetic Evolution Sandbox
#
# A tool for breeding audio-reactive cellular automata.
#
# HOW IT WORKS:
# 1. A population of 16 different cellular automata (CAs) is displayed. Each has its own unique "DNA" (rules).
# 2. The program plays a music file and analyzes it in real-time.
# 3. The music directly influences the CAs:
#    - BASS makes the patterns more stable.
#    - TREBLE makes them more chaotic and noisy.
#    - MIDS make the survival rules more lenient.
#    - BEATS trigger mutations, driving evolution.
# 4. YOU are the selector. Click on the CAs that produce interesting patterns. A blue border will appear.
# 5. Click the "Next Generation" button. The un-selected CAs die off. The selected ones "breed" to create
#    a new generation of offspring, which inherit and mutate the DNA of their parents.
# 6. Over time, you evolve beautiful, complex patterns that are uniquely synchronized to the music.
#
# SETUP:
# 1. You must have the following libraries installed:
#    pip install pygame numpy librosa
#
# 2. Place a music file ('.wav', '.flac', '.mp3') in the same directory as this script.
# 3. Change the `SONG_FILE` variable below to match your music file's name.
#
# --- END README ---

import pygame
import numpy as np
import librosa
import random

# --- CONFIGURATION ---
SONG_FILE = 'Singularity Anthem.wav'  # <--- CHANGE THIS TO YOUR MUSIC FILE
GRID_DIMENSIONS = (4, 4)     # A 4x4 grid of CAs
FPS = 30

# --- AESTHETIC & SIMULATION CONSTANTS ---
MAX_STATE = 5
DEAD_COLOR = np.array([10, 5, 15])      # A deep, dark purple instead of pure black
FADE_SPEED = 0.15                         # How quickly cells fade in and out
NOISE_BASE = 0.001
NOISE_TREBLE_SCALAR = 0.02                # How much treble affects chaos

# --- DERIVED CONSTANTS ---
POPULATION_SIZE = GRID_DIMENSIONS[0] * GRID_DIMENSIONS[1]
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800 # Keep screen square for simplicity
CELL_WIDTH = SCREEN_WIDTH // GRID_DIMENSIONS[0]
CELL_HEIGHT = SCREEN_HEIGHT // GRID_DIMENSIONS[1]
CA_COLS = CELL_WIDTH // 10
CA_ROWS = CELL_HEIGHT // 10

# --- COLOR PALETTES (will be mutated) ---
BASE_PALETTE = [
    DEAD_COLOR,
    (255, 50, 50), (50, 255, 50), (50, 50, 255),
    (255, 255, 50), (50, 255, 255)
]

# --- HELPER FUNCTIONS ---
def generate_random_dna():
    """Creates a dictionary representing the rules for a CA."""
    dna = {}
    for i in range(1, MAX_STATE + 1):
        dna[i] = {
            'birth': sorted(random.sample(range(1, 9), k=random.randint(1, 3))),
            'survival': sorted(random.sample(range(1, 9), k=random.randint(1, 4))),
            'spawn_state': random.randint(1, MAX_STATE)
        }
    return dna

def generate_random_palette():
    """Creates a vibrant, random color palette."""
    palette = [DEAD_COLOR]
    for _ in range(MAX_STATE):
        palette.append(np.random.randint(50, 256, size=3))
    return [np.array(c) for c in palette]

# --- CORE CLASSES ---

class MusicAnalyzer:
    """Pre-analyzes a song to provide real-time data efficiently."""
    def __init__(self, song_path):
        print(f"Loading and analyzing '{song_path}'...")
        try:
            y, self.sr = librosa.load(song_path)
            
            # Get beat timings
            tempo, self.beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
            self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
            self.next_beat_idx = 0

            # Get frequency content over time
            stft = librosa.stft(y)
            self.stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            self.times = librosa.times_like(self.stft_db)

            # Define frequency bands
            freqs = librosa.fft_frequencies(sr=self.sr)
            self.bass_bins = (freqs > 20) & (freqs < 250)
            self.mid_bins = (freqs >= 250) & (freqs < 4000)
            self.treble_bins = (freqs >= 4000) & (freqs < 20000)
            print("Analysis complete.")
        except Exception as e:
            print(f"Error loading song: {e}")
            print("Please ensure the song file is correct and readable.")
            pygame.quit()
            exit()
            
    def get_data_at_time(self, t):
        """Gets audio features for a specific time 't' in seconds."""
        # Find the closest analysis frame
        frame_idx = np.argmin(np.abs(self.times - t))
        
        # Normalize function for mapping [-80, 0] dB to [0, 1]
        def normalize_db(val):
            return (np.clip(val, -60, 0) + 60) / 60

        # Calculate energy in each band
        frame_data = self.stft_db[:, frame_idx]
        bass_energy = normalize_db(np.mean(frame_data[self.bass_bins]))
        mid_energy = normalize_db(np.mean(frame_data[self.mid_bins]))
        treble_energy = normalize_db(np.mean(frame_data[self.treble_bins]))
        
        # Check for a beat event
        is_beat = False
        if self.next_beat_idx < len(self.beat_times) and t >= self.beat_times[self.next_beat_idx]:
            is_beat = True
            self.next_beat_idx += 1
            
        return {
            'bass': bass_energy,
            'mid': mid_energy,
            'treble': treble_energy,
            'is_beat': is_beat
        }

class CA:
    """A single Cellular Automaton instance with its own DNA and state."""
    def __init__(self, size, dna, palette):
        self.rows, self.cols = size
        self.dna = dna
        self.palette = palette
        self.color_map = np.array(self.palette)

        self.state_grid = np.random.randint(0, MAX_STATE + 1, size=(self.rows, self.cols), dtype=np.int8)
        self.life_grid = self.state_grid.astype(np.float32)

    def update(self, music_params):
        # Apply environmental pressures from music
        noise = NOISE_BASE + music_params['treble'] * NOISE_TREBLE_SCALAR
        survival_leeway = 1 if music_params['mid'] > 0.6 else 0
        
        # Heavy bass stabilizes patterns by reducing noise
        if music_params['bass'] > 0.7:
            noise /= (1 + music_params['bass'])

        # --- Standard CA Logic ---
        is_alive = self.state_grid > 0
        neighbors = sum(np.roll(np.roll(is_alive, i, 0), j, 1)
                        for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0))
        
        # Create masks for birth and survival
        birth_mask = (self.state_grid == 0)
        survival_mask = (self.state_grid > 0)

        new_states = self.state_grid.copy()
        
        # Apply survival rules
        for state in range(1, MAX_STATE + 1):
            is_this_state = (self.state_grid == state)
            survival_rules = self.dna[state]['survival']
            
            # Check if neighbor count is within the survival range (with leeway)
            survives = np.isin(neighbors, [r + d for r in survival_rules for d in range(-survival_leeway, survival_leeway + 1)])
            
            # Cells of this state die if they don't meet survival criteria
            new_states[is_this_state & ~survives] = 0

        # Apply birth rules (vectorized for performance)
        # For simplicity, we'll let the most common neighboring parent dictate the new state
        parent_grid = np.zeros_like(self.state_grid)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i, j) != (0, 0):
                    rolled = np.roll(np.roll(self.state_grid, i, 0), j, 1)
                    # Simple majority rule (can be improved, but fast)
                    parent_grid[rolled > parent_grid] = rolled[rolled > parent_grid]

        for p_state in range(1, MAX_STATE + 1):
            is_parent = (parent_grid == p_state)
            birth_rules = self.dna[p_state]['birth']
            can_be_born = np.isin(neighbors, birth_rules)
            new_states[birth_mask & is_parent & can_be_born] = self.dna[p_state]['spawn_state']
        
        # Apply noise
        noise_mask = np.random.rand(self.rows, self.cols) < noise
        new_states[noise_mask] = np.random.randint(0, MAX_STATE + 1, size=1)

        self.state_grid = new_states
        
        # --- Update Life Grid for Fading Effect ---
        is_born_or_surviving = self.state_grid > 0
        self.life_grid = np.clip(self.life_grid + np.where(is_born_or_surviving, FADE_SPEED, -FADE_SPEED), 0, 1)

    def draw(self, surface):
        """Draws the CA grid to a given surface using fast numpy operations."""
        # Get the color for each cell based on its state
        alive_colors = self.color_map[self.state_grid]
        
        # Interpolate between DEAD_COLOR and ALIVE_COLOR based on 'life'
        # Reshape life_grid for broadcasting: (H, W) -> (H, W, 1)
        life = self.life_grid[:, :, np.newaxis]
        final_colors = (DEAD_COLOR * (1 - life) + alive_colors * life).astype(np.uint8)

        # Create a pygame surface from the color array and draw it
        temp_surface = pygame.surfarray.make_surface(np.transpose(final_colors, (1, 0, 2)))
        pygame.transform.scale(temp_surface, (surface.get_width(), surface.get_height()), surface)


class EvolutionManager:
    """Manages the population, selection, breeding, and mutation of CAs."""
    def __init__(self):
        self.population = [self.create_random_individual() for _ in range(POPULATION_SIZE)]
        self.generation = 1
        
    def create_random_individual(self):
        return CA((CA_ROWS, CA_COLS), generate_random_dna(), generate_random_palette())

    def update_population(self, music_params):
        # On beat, mutate a random individual
        if music_params['is_beat']:
            idx_to_mutate = random.randint(0, POPULATION_SIZE - 1)
            self.population[idx_to_mutate].dna = self.mutate(self.population[idx_to_mutate].dna)
            self.population[idx_to_mutate].palette = self.mutate_palette(self.population[idx_to_mutate].palette)

        for ca in self.population:
            ca.update(music_params)

    def mutate(self, dna):
        """Makes a small, random change to a DNA dictionary."""
        mutant_dna = {k: v.copy() for k,v in dna.items()} # Deep copy
        state_to_mutate = random.randint(1, MAX_STATE)
        rule_type = random.choice(['birth', 'survival', 'spawn_state'])
        
        if rule_type == 'spawn_state':
            mutant_dna[state_to_mutate]['spawn_state'] = random.randint(1, MAX_STATE)
        else:
            current_rules = mutant_dna[state_to_mutate][rule_type]
            if len(current_rules) > 1 and random.random() < 0.5:
                # Remove a rule
                current_rules.pop(random.randrange(len(current_rules)))
            else:
                # Add a new rule
                new_rule = random.randint(1, 8)
                if new_rule not in current_rules:
                    current_rules.append(new_rule)
            mutant_dna[state_to_mutate][rule_type] = sorted(current_rules)
        return mutant_dna
        
    def mutate_palette(self, palette):
        """Slightly alters one color in a palette."""
        mutant_palette = [c.copy() for c in palette]
        idx = random.randint(1, MAX_STATE) # Don't mutate dead color
        change = np.random.randint(-20, 21, size=3)
        mutant_palette[idx] = np.clip(mutant_palette[idx] + change, 0, 255)
        return mutant_palette

    def breed(self, dna1, dna2):
        """Combines the DNA of two parents using crossover."""
        child_dna = {}
        for state in range(1, MAX_STATE + 1):
            # Crossover on a per-state basis
            child_dna[state] = random.choice([dna1[state], dna2[state]]).copy()
        return child_dna

    def next_generation(self, selected_indices):
        """Creates a new population by breeding selected individuals."""
        if not selected_indices:
            print("No individuals selected. The apocalypse is upon us.")
            self.population = [self.create_random_individual() for _ in range(POPULATION_SIZE)]
            self.generation = 1
            return
            
        self.generation += 1
        print(f"Breeding Generation {self.generation}...")
        
        selected_parents = [self.population[i] for i in selected_indices]
        new_population = []
        
        for _ in range(POPULATION_SIZE):
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)
            
            # Breed and mutate
            child_dna = self.breed(parent1.dna, parent2.dna)
            child_dna = self.mutate(child_dna)

            child_palette = [c.copy() for c in random.choice([parent1.palette, parent2.palette])]
            child_palette = self.mutate_palette(child_palette)
            
            new_population.append(CA((CA_ROWS, CA_COLS), child_dna, child_palette))
        
        self.population = new_population

class Game:
    """The main application class that ties everything together."""
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Synesthetic Evolution Sandbox")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sans-serif", 24)
        self.font_small = pygame.font.SysFont("sans-serif", 16)

        self.music = MusicAnalyzer(SONG_FILE)
        self.evo_manager = EvolutionManager()
        
        self.selected_indices = []

        # Create surfaces for each CA cell
        self.ca_surfaces = [pygame.Surface((CELL_WIDTH, CELL_HEIGHT)) for _ in range(POPULATION_SIZE)]

    def run(self):
        # Start music playback
        pygame.mixer.music.load(SONG_FILE)
        pygame.mixer.music.play(-1) # Loop forever
        
        running = True
        while running:
            music_time = pygame.mixer.music.get_pos() / 1000.0 # Time in seconds
            music_data = self.music.get_data_at_time(music_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: # Spacebar for Next Gen
                        self.evo_manager.next_generation(self.selected_indices)
                        self.selected_indices = []
            
            self.update(music_data)
            self.draw(music_data)
            
            self.clock.tick(FPS)
            pygame.display.set_caption(f"Synesthetic Sandbox - Gen {self.evo_manager.generation} - FPS: {self.clock.get_fps():.1f}")
            
        pygame.quit()

    def handle_click(self, pos):
        # Check for button click first
        if self.next_gen_button_rect.collidepoint(pos):
             self.evo_manager.next_generation(self.selected_indices)
             self.selected_indices = []
             return

        # Then check for CA grid click
        grid_x = pos[0] // CELL_WIDTH
        grid_y = pos[1] // CELL_HEIGHT
        index = grid_y * GRID_DIMENSIONS[0] + grid_x
        
        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.append(index)

    def update(self, music_data):
        self.evo_manager.update_population(music_data)

    def draw(self, music_data):
        self.screen.fill((0, 0, 0))

        # Draw each CA in the population
        for i, ca in enumerate(self.evo_manager.population):
            row, col = divmod(i, GRID_DIMENSIONS[0])
            target_rect = (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            ca.draw(self.ca_surfaces[i])
            self.screen.blit(self.ca_surfaces[i], target_rect)
            
            # Draw selection border
            if i in self.selected_indices:
                pygame.draw.rect(self.screen, (50, 150, 255), target_rect, 3)

        # --- Draw Global "Glow" Effect ---
        glow_alpha = music_data['bass'] * 100 + music_data['mid'] * 50
        glow_color = (
            int(music_data['bass'] * 200),
            int(music_data['mid'] * 100),
            int(music_data['treble'] * 255)
        )
        glow_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        glow_surface.fill((*glow_color, np.clip(glow_alpha, 0, 100)))
        self.screen.blit(glow_surface, (0, 0))

        # --- Draw UI Elements ---
        # "Next Generation" button
        self.next_gen_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 60, 200, 40)
        pygame.draw.rect(self.screen, (200, 200, 200), self.next_gen_button_rect)
        btn_text = self.font.render("Next Generation", True, (0, 0, 0))
        self.screen.blit(btn_text, btn_text.get_rect(center=self.next_gen_button_rect.center))

        # Instructions
        instr_text = self.font_small.render("Click to select parents. Press SPACE or click button to breed.", True, (200, 200, 200))
        self.screen.blit(instr_text, instr_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80)))
        
        pygame.display.flip()

if __name__ == '__main__':
    Game().run()
