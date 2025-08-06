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
    "DRONE_A": (0, 0, 255),
    "DRONE_B": (255, 0, 255),
    "FOV_OUTLINE": (0, 255, 100),
    "PATH": (200, 200, 0),
}

class Drone:
    def __init__(self, x, y, name="A"):
        self.x = x
        self.y = y
        self.name = name
        self.direction = (0, 0)

class DroneJammerEnv:
    def __init__(self):
        self.terrain_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.belief_grid = np.full((GRID_HEIGHT, GRID_WIDTH), -1)
        self.drone1 = None
        self.drone2 = None
        self.goal1 = None
        self.goal2 = None
        self.done1 = False
        self.done2 = False
        self.path1 = []
        self.path2 = []
        self.randomize_start_and_goals(min_distance=15)
        self.place_objects()

    def randomize_start_and_goals(self, min_distance=15):
        while True:
            start_x, start_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            goal1_x, goal1_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            goal2_x, goal2_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            d1 = abs(start_x - goal1_x) + abs(start_y - goal1_y)
            d2 = abs(start_x - goal2_x) + abs(start_y - goal2_y)
            if d1 >= min_distance and d2 >= min_distance and (goal1_x, goal1_y) != (goal2_x, goal2_y):
                self.drone1 = Drone(start_x, start_y, "A")
                self.drone2 = Drone(start_x, start_y, "B")
                self.goal1 = (goal1_x, goal1_y)
                self.goal2 = (goal2_x, goal2_y)
                break

    def place_objects(self):
        self.terrain_grid[...] = EMPTY
        self.terrain_grid[self.goal1[1], self.goal1[0]] = GOAL
        self.terrain_grid[self.goal2[1], self.goal2[0]] = GOAL
        for _ in range(150):
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in [(self.drone1.x, self.drone1.y), self.goal1, self.goal2]:
                self.terrain_grid[y, x] = OBSTACLE
        for _ in range(30):
            x = random.randint(0, GRID_WIDTH - 2)
            y = random.randint(0, GRID_HEIGHT - 2)
            for dx in range(2):
                for dy in range(2):
                    if (x+dx, y+dy) not in [(self.drone1.x, self.drone1.y), self.goal1, self.goal2]:
                        self.terrain_grid[y+dy, x+dx] = JAMMER

    def get_fov(self, drone):
        x, y = drone.x, drone.y
        xmin, xmax = max(0, x - 2), min(GRID_WIDTH, x + 3)
        ymin, ymax = max(0, y - 2), min(GRID_HEIGHT, y + 3)
        fov = self.terrain_grid[ymin:ymax, xmin:xmax].copy()
        if self.terrain_grid[y, x] == JAMMER:
            fov[...] = -1
        return fov, (xmin, ymin)

    def update_belief(self, drone):
        fov, (xmin, ymin) = self.get_fov(drone)
        if self.terrain_grid[drone.y, drone.x] != JAMMER:
            self.belief_grid[ymin:ymin+fov.shape[0], xmin:xmin+fov.shape[1]] = fov

    def move_drone(self, drone, dx, dy):
        new_x = max(0, min(GRID_WIDTH - 1, drone.x + dx))
        new_y = max(0, min(GRID_HEIGHT - 1, drone.y + dy))
        if self.terrain_grid[new_y, new_x] != OBSTACLE:
            drone.x, drone.y = new_x, new_y
            drone.direction = (dx, dy)

    def a_star_path(self, drone, goal, other_drone_pos):
        start = (drone.x, drone.y)
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
                    if val == OBSTACLE or neighbor == other_drone_pos:
                        continue
                    move_cost = 1 if val != JAMMER else 3
                    tentative_g = g_score[current] + move_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
                        came_from[neighbor] = current
        return []

# Visualization Setup
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Two Drones with Jammer-Aware A*")
clock = pygame.time.Clock()
env = DroneJammerEnv()
frames = []

def draw_grid():
    screen.fill((255, 255, 255))
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            val = env.terrain_grid[y, x]
            pygame.draw.rect(screen, COLORS[val], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            if env.belief_grid[y, x] == -1:
                pygame.draw.rect(screen, COLORS["FOV_MASK"], rect)

    for px, py in env.path1 + env.path2:
        rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLORS["PATH"], rect)

    rect_a = pygame.Rect(env.drone1.x * CELL_SIZE, env.drone1.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.circle(screen, COLORS["DRONE_A"], rect_a.center, CELL_SIZE // 3)
    rect_b = pygame.Rect(env.drone2.x * CELL_SIZE, env.drone2.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.circle(screen, COLORS["DRONE_B"], rect_b.center, CELL_SIZE // 3)

    pygame.display.flip()

# Main loop
running = True
while running:
    pygame.event.pump()
    env.update_belief(env.drone1)
    env.update_belief(env.drone2)
    env.path1 = env.a_star_path(env.drone1, env.goal1, (env.drone2.x, env.drone2.y))
    env.path2 = env.a_star_path(env.drone2, env.goal2, (env.drone1.x, env.drone1.y))

    draw_grid()
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame.copy())

    if not env.done1 and env.path1:
        dx, dy = env.path1[0][0] - env.drone1.x, env.path1[0][1] - env.drone1.y
        if (env.drone1.x + dx, env.drone1.y + dy) != (env.drone2.x, env.drone2.y):
            env.move_drone(env.drone1, dx, dy)
        if (env.drone1.x, env.drone1.y) == env.goal1:
            print("Drone A reached its goal!")
            env.done1 = True

    if not env.done2 and env.path2:
        dx, dy = env.path2[0][0] - env.drone2.x, env.path2[0][1] - env.drone2.y
        if (env.drone2.x + dx, env.drone2.y + dy) != (env.drone1.x, env.drone1.y):
            env.move_drone(env.drone2, dx, dy)
        if (env.drone2.x, env.drone2.y) == env.goal2:
            print("Drone B reached its goal!")
            env.done2 = True

    if env.done1 and env.done2:
        running = False

    clock.tick(10)

# Save Video
output_path = "two_drones_a_star_run.mp4"
imageio.mimsave(output_path, frames, fps=4)
pygame.quit()
print("Video saved to", output_path)
