import pygame
import sys
import numpy as np

# --- Constants ---
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 4
COLUMNS = WIDTH // CELL_SIZE
ROWS = HEIGHT // CELL_SIZE
FPS = 60 # Increased slightly due to performance gains

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (50, 50, 50)
BUTTON_HOVER_COLOR = (70, 70, 70)
SLIDER_BG_COLOR = (50, 50, 50)
SLIDER_HANDLE_COLOR = (100, 100, 100)


# --- UI Element Classes (Button, Slider) ---
# These classes are well-structured and remain unchanged.
class Button:
    """A clickable button."""
    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.font = pygame.font.SysFont(None, 24)
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
        elif event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        text_surf = self.font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

class Slider:
    """A slider to control a value."""
    def __init__(self, rect, min_val, max_val, initial_val, callback):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.callback = callback
        self.value = initial_val
        self.handle_rect = pygame.Rect(0, 0, 20, self.rect.height)
        self.update_handle_pos()
        self.is_dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) or self.handle_rect.collidepoint(event.pos):
                self.is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self.update_value_from_mouse(event.pos[0])

    def update_value_from_mouse(self, mouse_x):
        x = max(self.rect.left, min(self.rect.right - self.handle_rect.width, mouse_x))
        self.handle_rect.x = x
        self.value = self.min_val + (self.handle_rect.x - self.rect.left) / (self.rect.width - self.handle_rect.width) * (self.max_val - self.min_val)
        self.callback(self.value)

    def update_handle_pos(self):
        self.handle_rect.centery = self.rect.centery
        handle_x = self.rect.left + (self.value - self.min_val) / (self.max_val - self.min_val) * (self.rect.width - self.handle_rect.width)
        self.handle_rect.x = handle_x

    def draw(self, screen):
        pygame.draw.rect(screen, SLIDER_BG_COLOR, self.rect)
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, self.handle_rect)


# --- Cellular Automata Class ---
class CA:
    """Manages the CA grid and logic. Is now independent of global rules."""
    def __init__(self, width, height, dna, palette):
        self.width = width
        self.height = height
        
        # IMPROVEMENT: Dependency Injection. Rules are passed in, not global.
        self.dna = dna
        self.palette = palette
        self.color_map = np.array(self.palette, dtype=np.uint8)
        self.max_state = len(self.palette) - 1
        
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.running = False
        self.noise_probability = 0.0

    def initialize_random(self):
        self.grid = np.random.randint(0, self.max_state + 1, size=(self.height, self.width), dtype=np.int8)

    def update(self):
        if not self.running:
            return

        new_grid = self.grid.copy()
        
        is_alive = self.grid > 0
        neighbors = sum(np.roll(np.roll(is_alive, i, 0), j, 1)
                        for i in [-1, 0, 1] for j in [-1, 0, 1]
                        if (i, j) != (0, 0))

        for i in range(self.height):
            for j in range(self.width):
                current_state = self.grid[i, j]
                num_neighbors = neighbors[i, j]

                if current_state > 0:  # Survival logic for living cells
                    if num_neighbors not in self.dna[current_state]['survival']:
                        new_grid[i, j] = 0
                else:  # Birth logic for dead cells
                    # IMPROVEMENT: Fixes directional bias by choosing a random parent.
                    possible_parents = []
                    for x_offset in [-1, 0, 1]:
                        for y_offset in [-1, 0, 1]:
                            if x_offset == 0 and y_offset == 0:
                                continue
                            
                            parent_state = self.grid[(i + x_offset) % self.height, (j + y_offset) % self.width]
                            if parent_state > 0 and num_neighbors in self.dna[parent_state]['birth']:
                                possible_parents.append(parent_state)
                    
                    if possible_parents:
                        # Randomly select one of the valid parents
                        chosen_parent = np.random.choice(possible_parents)
                        new_grid[i, j] = self.dna[chosen_parent]['spawn_state']

        if self.noise_probability > 0:
            noise_mask = np.random.rand(self.height, self.width) < self.noise_probability
            new_grid[noise_mask] = np.random.randint(0, self.max_state + 1, size=new_grid.shape)[noise_mask]

        self.grid = new_grid

    def draw_grid(self, screen):
        """IMPROVEMENT: This method is now dramatically faster."""
        try:
            # 1. Create an array of colors corresponding to the grid states.
            color_grid = self.color_map[self.grid]
            
            # 2. Create a surface directly from the NumPy array of colors.
            #    We must transpose the array from (height, width, 3) to (width, height, 3) for pygame.
            surface = pygame.surfarray.make_surface(np.transpose(color_grid, (1, 0, 2)))
            
            # 3. Scale the small pixel surface up to the full window size.
            scaled_surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))
            
            # 4. Blit the single, scaled surface to the screen.
            screen.blit(scaled_surface, (0, 0))
        except IndexError:
            # A failsafe in case a state in the grid is out of bounds for the palette.
            print("Error: A cell state is outside the palette range. Re-initializing.")
            self.initialize_random()

# --- Main Game Class ---
class Game:
    """Main class to run the simulation."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Evolving CA Biom Simulation (Refined)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # IMPROVEMENT: Rules are now defined within the Game scope.
        # This makes the CA class reusable and the logic easier to find.
        
        # --- The Palette ---
        # State 0 is always dead (BLACK). Add or remove colors here.
        palette = [
            (0, 0, 0),        # State 0: Dead
            (200, 30, 30),    # State 1: A "species" that likes to be sparse.
            (30, 200, 30),    # State 2: A colonial species.
            (30, 30, 200),    # State 3: A durable species.
            (200, 200, 30),   # State 4: A fast-spreading "mold".
            (200, 30, 200),   # State 5: A pattern-maker.
            (30, 200, 200),   # State 6: A chaotic species.
        ]

        # --- The DNA: The Rules of Life and Evolution ---
        # This is where you can "play god". Change these numbers to see what happens!
        # 'birth': [counts] -> A dead cell with a parent of this state will be born if it has this many *total* neighbors.
        # 'survival': [counts] -> A living cell of this state survives if it has this many neighbors.
        # 'spawn_state': state -> When a cell of this state becomes a parent, its child will be this new state.
        dna = {
            1: {'birth': [3],       'survival': [1, 2],     'spawn_state': 2},
            2: {'birth': [3],       'survival': [4, 5, 6],  'spawn_state': 3},
            3: {'birth': [2, 3],    'survival': [3, 4, 5],  'spawn_state': 4},
            4: {'birth': [3],       'survival': [2, 3],     'spawn_state': 5}, # Classic Conway's Game of Life rules
            5: {'birth': [4],       'survival': [3, 4, 5],  'spawn_state': 6},
            6: {'birth': [2, 3, 4], 'survival': [5, 6, 7],  'spawn_state': 1}, # Creates a cycle back to state 1
        }

        self.ca = CA(COLUMNS, ROWS, dna, palette)
        self.ca.initialize_random()

        self.setup_ui()

    def setup_ui(self):
        self.buttons = [
            Button((10, 10, 100, 40), "Pause/Resume", self.toggle_running),
            Button((120, 10, 100, 40), "Reset", self.ca.initialize_random),
            Button((230, 10, 100, 40), "Step", self.ca.update)
        ]

        # IMPROVEMENT: Increased slider range for more chaotic possibilities.
        self.slider = Slider((10, 60, 200, 20), 0.0, 0.05, 0.0, self.set_noise_probability)
        self.noise_label = self.font.render(f"Noise: {self.ca.noise_probability:.4f}", True, WHITE)

    def toggle_running(self):
        self.ca.running = not self.ca.running

    def set_noise_probability(self, value):
        self.ca.noise_probability = value
        self.noise_label = self.font.render(f"Noise: {self.ca.noise_probability:.4f}", True, WHITE)

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for button in self.buttons:
                button.handle_event(event)
            self.slider.handle_event(event)

    def update(self):
        self.ca.update()

    def draw(self):
        self.screen.fill(BLACK)
        self.ca.draw_grid(self.screen)
        
        # Draw UI elements on top of the simulation
        for button in self.buttons:
            button.draw(self.screen)
        self.slider.draw(self.screen)
        self.screen.blit(self.noise_label, (self.slider.rect.right + 10, self.slider.rect.centery - self.noise_label.get_height() // 2))
        
        pygame.display.flip()

if __name__ == '__main__':
    game = Game()
    game.run()
