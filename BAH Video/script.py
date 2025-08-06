import pygame
import numpy as np
import random
import imageio
import heapq

# Constants
GRID_WIDTH = 25
GRID_HEIGHT = 25
CELL_SIZE = int(800 / GRID_WIDTH)

# Object types
EMPTY = 0
OBSTACLE = 1
GOAL = 2
JAMMER = 3

# Colors
COLORS = {
    EMPTY: (255, 255, 255),
    OBSTACLE: (0, 0, 0),
    GOAL: (0, 255, 0),
    JAMMER: (255, 0, 0),
    "FOV_MASK": (180, 180, 180),
    "DRONE": (0, 0, 255),
    "FOV_OUTLINE": (0, 255, 100),
    "PATH": (200, 200, 0),
}

class Drone:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = (0, 0)

class DroneJammerEnv:
    def __init__(self):
        self.terrain_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.belief_grid = np.full((GRID_HEIGHT, GRID_WIDTH), -1)
        self.drone = None
        self.goal_pos = None
        self.done = False
        self.path = []
        self.randomize_start_and_goal(min_distance=15)
        self.place_objects()

    def randomize_start_and_goal(self, min_distance=15):
        while True:
            start_x, start_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            goal_x, goal_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            dist = abs(start_x - goal_x) + abs(start_y - goal_y)
            if dist >= min_distance:
                self.drone = Drone(start_x, start_y)
                self.goal_pos = (goal_x, goal_y)
                break

    def place_objects(self):
        self.terrain_grid[...] = EMPTY
        self.terrain_grid[self.goal_pos[1], self.goal_pos[0]] = GOAL
        for _ in range(150):
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in [(self.drone.x, self.drone.y), self.goal_pos]:
                self.terrain_grid[y, x] = OBSTACLE
        for _ in range(20):
            x = random.randint(0, GRID_WIDTH - 2)
            y = random.randint(0, GRID_HEIGHT - 2)
            for dx in range(2):
                for dy in range(2):
                    if (x+dx, y+dy) not in [(self.drone.x, self.drone.y), self.goal_pos]:
                        self.terrain_grid[y+dy, x+dx] = JAMMER

    def get_fov(self):
        x, y = self.drone.x, self.drone.y
        xmin, xmax = max(0, x - 2), min(GRID_WIDTH, x + 3)
        ymin, ymax = max(0, y - 2), min(GRID_HEIGHT, y + 3)
        fov = self.terrain_grid[ymin:ymax, xmin:xmax].copy()
        if self.terrain_grid[y, x] == JAMMER:
            fov[...] = -1
        return fov, (xmin, ymin)

    def update_belief(self):
        fov, (xmin, ymin) = self.get_fov()
        if self.terrain_grid[self.drone.y, self.drone.x] != JAMMER:
            self.belief_grid[ymin:ymin+fov.shape[0], xmin:xmin+fov.shape[1]] = fov

    def move_drone(self, dx, dy):
        if self.done:
            return
        new_x = max(0, min(GRID_WIDTH - 1, self.drone.x + dx))
        new_y = max(0, min(GRID_HEIGHT - 1, self.drone.y + dy))
        if self.terrain_grid[new_y, new_x] != OBSTACLE:
            self.drone.x, self.drone.y = new_x, new_y
            self.drone.direction = (dx, dy)
            if (new_x, new_y) == self.goal_pos:
                print("Drone reached the goal!")
                self.done = True

    def a_star_path(self):
        start = (self.drone.x, self.drone.y)
        goal = self.goal_pos
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < GRID_WIDTH and 0 <= neighbor[1] < GRID_HEIGHT:
                    val = self.belief_grid[neighbor[1], neighbor[0]]
                    if val == OBSTACLE:
                        continue
                    move_cost = 1 if val != JAMMER else 3
                    tentative_g = g_score[current] + move_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
                        came_from[neighbor] = current
        return []

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("FOV-Aware A* Drone Navigation (Jam-Aware)")
clock = pygame.time.Clock()
env = DroneJammerEnv()
frames = []

def draw_grid():
    screen.fill((255, 255, 255))
    fov, (xmin, ymin) = env.get_fov()

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            val = env.terrain_grid[y, x]
            pygame.draw.rect(screen, COLORS[val], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if env.belief_grid[y, x] == -1:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLORS["FOV_MASK"], rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

    for px, py in env.path:
        rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLORS["PATH"], rect)

    if fov[2, 2] != -1:
        fov_rect = pygame.Rect(xmin * CELL_SIZE, ymin * CELL_SIZE, fov.shape[1] * CELL_SIZE, fov.shape[0] * CELL_SIZE)
        pygame.draw.rect(screen, COLORS["FOV_OUTLINE"], fov_rect, 5)

    drone_rect = pygame.Rect(env.drone.x * CELL_SIZE, env.drone.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.circle(screen, COLORS["DRONE"], drone_rect.center, CELL_SIZE // 3)

    pygame.display.flip()

# Main loop
running = True
while running:
    pygame.event.pump()
    env.update_belief()
    env.path = env.a_star_path()
    draw_grid()

    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame.copy())

    if env.done or not env.path:
        running = False
        continue

    next_pos = env.path[0]
    dx = next_pos[0] - env.drone.x
    dy = next_pos[1] - env.drone.y
    env.move_drone(dx, dy)
    clock.tick(10)

# Save video
output_path = "drone_a_star_run.mp4"
print(f"Saving video to {output_path}...")
imageio.mimsave(output_path, frames, fps=4)
pygame.quit()
print("Video saved successfully.")