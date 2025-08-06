import pygame
import numpy as np
import random
import imageio
import heapq

# Constants
GRID_WIDTH = 25
GRID_HEIGHT = 25
CELL_SIZE = int(800 / GRID_WIDTH)

EMPTY = 0
OBSTACLE = 1
GOAL = 2

COLORS = {
    EMPTY: (255, 255, 255),
    OBSTACLE: (0, 0, 0),
    GOAL: (0, 255, 0),
    "FOV_MASK": (180, 180, 180),
    "DRONE_A": (0, 0, 255),
    "DRONE_B": (255, 0, 255),
    "FOV_OUTLINE": (0, 255, 100),
    "PATH": (200, 200, 0),
    "SURVEILLANCE": (255, 165, 0),
}

class Drone:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
        self.direction = (0, 0)

class DroneJammerEnv:
    def __init__(self):
        self.terrain_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.belief_grid = np.full((GRID_HEIGHT, GRID_WIDTH), -1)
        self.drone1 = self.drone2 = None
        self.goal1 = self.goal2 = None
        self.done1 = self.done2 = False
        self.path1 = self.path2 = []
        self.surveillance_zones = []
        self.randomize_start_and_goals()
        self.place_objects()

    def randomize_start_and_goals(self, min_distance=15):
        while True:
            sx, sy = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            gx1, gy1 = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            gx2, gy2 = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            d1 = abs(sx - gx1) + abs(sy - gy1)
            d2 = abs(sx - gx2) + abs(sy - gy2)
            if d1 >= min_distance and d2 >= min_distance and (gx1, gy1) != (gx2, gy2):
                self.drone1 = Drone(sx, sy, "A")
                self.drone2 = Drone(sx, sy, "B")
                self.goal1 = (gx1, gy1)
                self.goal2 = (gx2, gy2)
                break

    def place_objects(self):
        self.terrain_grid[...] = EMPTY
        self.terrain_grid[self.goal1[1], self.goal1[0]] = GOAL
        self.terrain_grid[self.goal2[1], self.goal2[0]] = GOAL
        for _ in range(150):
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in [(self.drone1.x, self.drone1.y), self.goal1, self.goal2]:
                self.terrain_grid[y, x] = OBSTACLE

    def update_surveillance_zones(self):
        new_zones = []
        for zone in self.surveillance_zones:
            zone["duration"] -= 1
            if zone["duration"] > 0:
                new_zones.append(zone)
        for _ in range(random.randint(1, 4)):
            tries = 0
            while tries < 50:
                x, y = random.randint(0, GRID_WIDTH - 3), random.randint(0, GRID_HEIGHT - 3)
                zone_cells = [(x + dx, y + dy) for dx in range(2) for dy in range(2)]
                if all(cell not in [self.goal1, self.goal2] for cell in zone_cells):
                    new_zones.append({"x": x, "y": y, "duration": random.randint(5, 15)})
                    break
                tries += 1

        self.surveillance_zones = new_zones

    def in_surveillance(self, pos):
        x, y = pos
        for zone in self.surveillance_zones:
            if zone["x"] <= x < zone["x"] + 2 and zone["y"] <= y < zone["y"] + 2:
                return True
        return False

    def in_belief_surveillance(self, pos):
        x, y = pos
        if self.belief_grid[y, x] == -1:
            return False  # Unknown area treated as safe
        return self.in_surveillance(pos)  # Use global info to check surveillance, but only if it's visible


    def get_fov(self, drone):
        x, y = drone.x, drone.y
        xmin, xmax = max(0, x - 2), min(GRID_WIDTH, x + 3)
        ymin, ymax = max(0, y - 2), min(GRID_HEIGHT, y + 3)
        return self.terrain_grid[ymin:ymax, xmin:xmax].copy(), (xmin, ymin)

    def update_belief(self, drone):
        fov, (xmin, ymin) = self.get_fov(drone)
        self.belief_grid[ymin:ymin + fov.shape[0], xmin:xmin + fov.shape[1]] = fov



    def move_drone(self, drone, dx, dy):
        curr_pos = (drone.x, drone.y)
        new_pos = (max(0, min(GRID_WIDTH - 1, drone.x + dx)),
                   max(0, min(GRID_HEIGHT - 1, drone.y + dy)))
    
        # Allow movement if not into obstacle
        if self.terrain_grid[new_pos[1], new_pos[0]] == OBSTACLE:
            return
    
        # Only restrict entering surveillance zone from outside
        if not self.in_surveillance(curr_pos) and self.in_surveillance(new_pos):
            return
    
        drone.x, drone.y = new_pos
        drone.direction = (dx, dy)


    def a_star_path(self, drone, goal, other_drone_pos, allow_surveillance=False):
        start = (drone.x, drone.y)
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
    
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                    continue
                if neighbor == other_drone_pos:
                    continue
                if self.belief_grid[ny, nx] == OBSTACLE:
                    continue
                if not allow_surveillance and self.in_belief_surveillance(neighbor):
                    continue
                tentative = g_score[current] + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    heapq.heappush(open_set, (tentative + h(neighbor), neighbor))
                    came_from[neighbor] = current
        return []
    
    
    def decide_path(self, drone, goal, other_drone_pos):
        # Try shortest path (even if through surveillance)
        path_with_surv = self.a_star_path(drone, goal, other_drone_pos, allow_surveillance=True)
        blocked = any(self.in_surveillance(pos) for pos in path_with_surv[:3])  # check first 3 steps
    
        if not path_with_surv or not blocked:
            return path_with_surv  # safe path or not blocked at front
    
        # Estimate wait cost
        wait_cost = min([
    zone["duration"]
    for pos in path_with_surv[:3]
    for zone in self.surveillance_zones
    if zone["x"] <= pos[0] < zone["x"] + 2 and zone["y"] <= pos[1] < zone["y"] + 2
] or [10])

    
        # Try alternate route avoiding surveillance
        path_no_surv = self.a_star_path(drone, goal, other_drone_pos, allow_surveillance=False)
        reroute_cost = len(path_no_surv) if path_no_surv else float("inf")
    
        if wait_cost < reroute_cost:
            return []  # Wait at current location
        else:
            return path_no_surv



# Visualization
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Two Drones Avoiding Surveillance")
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

    for zone in env.surveillance_zones:
        for dx in range(2):
            for dy in range(2):
                zx, zy = zone["x"] + dx, zone["y"] + dy
                if 0 <= zx < GRID_WIDTH and 0 <= zy < GRID_HEIGHT:
                    if env.belief_grid[zy, zx] != -1:
                        rect = pygame.Rect(zx * CELL_SIZE, zy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, COLORS["SURVEILLANCE"], rect)

    for px, py in env.path1:
        rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (173, 216, 230), rect)  # Light blue for drone A

    for px, py in env.path2:
        rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 182, 193), rect)  # Light pink for drone B


    pygame.draw.circle(screen, COLORS["DRONE_A"], (env.drone1.x * CELL_SIZE + CELL_SIZE//2, env.drone1.y * CELL_SIZE + CELL_SIZE//2), CELL_SIZE // 3)
    pygame.draw.circle(screen, COLORS["DRONE_B"], (env.drone2.x * CELL_SIZE + CELL_SIZE//2, env.drone2.y * CELL_SIZE + CELL_SIZE//2), CELL_SIZE // 3)
    pygame.display.flip()




# Main simulation loop
running = True
while running:
    pygame.event.pump()
    env.update_surveillance_zones()
    env.update_belief(env.drone1)
    env.update_belief(env.drone2)

    env.path1 = env.decide_path(env.drone1, env.goal1, (env.drone2.x, env.drone2.y))
    env.path2 = env.decide_path(env.drone2, env.goal2, (env.drone1.x, env.drone1.y))


    draw_grid()
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame.copy())

    if not env.done1 and env.path1:
        next1 = env.path1[0]
        if next1 != (env.drone2.x, env.drone2.y):
            dx, dy = next1[0] - env.drone1.x, next1[1] - env.drone1.y
            env.move_drone(env.drone1, dx, dy)
        if (env.drone1.x, env.drone1.y) == env.goal1:
            print("Drone A reached its goal!")
            env.done1 = True
    
    if not env.done2 and env.path2:
        next2 = env.path2[0]
        if next2 != (env.drone1.x, env.drone1.y):
            dx, dy = next2[0] - env.drone2.x, next2[1] - env.drone2.y
            env.move_drone(env.drone2, dx, dy)
        if (env.drone2.x, env.drone2.y) == env.goal2:
            print("Drone B reached its goal!")
            env.done2 = True


    if env.done1 and env.done2:
        running = False

    clock.tick(10)

# Save the video
output_path = "two_drones_surveillance_run.mp4"
imageio.mimsave(output_path, frames, fps=3)
pygame.quit()
print("Video saved to", output_path)
