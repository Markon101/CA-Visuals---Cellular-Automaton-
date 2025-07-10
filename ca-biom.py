import pygame
import sys
import secrets
import numpy as np

# --- Constants ---
WIDTH, HEIGHT = 1080, 1920
CELL_SIZE = 10
COLUMNS = WIDTH // CELL_SIZE
ROWS = HEIGHT // CELL_SIZE
FPS = 30

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BUTTON_COLOR = (50, 50, 50)
BUTTON_HOVER_COLOR = (70, 70, 70)
SLIDER_BG_COLOR = (50, 50, 50)
SLIDER_HANDLE_COLOR = (100, 100, 100)

# --- UI Element Classes ---

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
            if self.rect.collidepoint(event.pos):
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
    """Manages the cellular automata grid and logic."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.running = False
        self.noise_probability = 0.0

    def initialize_random(self):
        self.grid = np.random.randint(2, size=(self.height, self.width), dtype=np.int8)

    def update(self):
        if not self.running:
            return

        # Using numpy for faster neighbor counting
        neighbors = sum(np.roll(np.roll(self.grid, i, 0), j, 1)
                        for i in [-1, 0, 1] for j in [-1, 0, 1]
                        if (i, j) != (0, 0))

        # Applying Conway's Game of Life rules
        born = (self.grid == 0) & (neighbors == 3)
        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        new_grid = np.zeros_like(self.grid)
        new_grid[born | survive] = 1

        # Apply noise
        if self.noise_probability > 0:
            noise_mask = np.random.rand(self.height, self.width) < self.noise_probability
            new_grid[noise_mask] = 1 - new_grid[noise_mask]

        self.grid = new_grid

    def draw_grid(self, screen):
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    pygame.draw.rect(screen, WHITE, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# --- Main Game Class ---

class Game:
    """Main class to run the simulation."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("CA Biom Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.ca = CA(COLUMNS, ROWS)
        self.ca.initialize_random()

        self.setup_ui()

    def setup_ui(self):
        self.buttons = [
            Button((10, 10, 100, 40), "Pause/Resume", self.toggle_running),
            Button((120, 10, 100, 40), "Reset", self.ca.initialize_random),
            Button((230, 10, 100, 40), "Step", self.ca.update)
        ]

        self.slider = Slider((10, 60, 200, 20), 0.0, 0.1, 0.0, self.set_noise_probability)
        self.noise_label = self.font.render(f"Noise: {self.ca.noise_probability:.3f}", True, WHITE)

    def toggle_running(self):
        self.ca.running = not self.ca.running

    def set_noise_probability(self, value):
        self.ca.noise_probability = value
        self.noise_label = self.font.render(f"Noise: {self.ca.noise_probability:.3f}", True, WHITE)

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
        for button in self.buttons:
            button.draw(self.screen)
        self.slider.draw(self.screen)
        self.screen.blit(self.noise_label, (self.slider.rect.right + 10, self.slider.rect.centery - self.noise_label.get_height() // 2))
        pygame.display.flip()

if __name__ == '__main__':
    game = Game()
    game.run()
