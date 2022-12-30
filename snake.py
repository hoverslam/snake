import numpy as np
import pygame
import sys


class Game:
    
    def __init__(self, size:tuple[int, int]=(40, 40)) -> None:
        self.size = size
        self.pitch = np.zeros(size, dtype=int)
        self.score = 0
        
        # Initialize snake with a centered starting position and a length of 3
        self.snake = Snake([(self.size[0] // 2, self.size[1] // 2 - i) for i in range(3)])
        
        # Initialize food
        self.food = Food()
        
    def step(self, action:int) -> tuple[np.ndarray, float, bool, bool, None]:
        reward = 0.0
        terminated = [False, False]
        
        # Change direction if action is not zero
        if action != 0:
            self.snake.change_direction(action)
               
        # Move snake
        tail, terminated[0] = self.snake.move(self.size)

        # If snake has bitten its own body the game is over
        terminated[1] = self.snake.positions[0] in self.snake.positions[1:]
        
        # Only move on if the game is not terminated
        if not any(terminated):        
            # If snake has eaten the food generate new one and increase body length
            if self.snake.positions[0] == self.food.position:
                self.food.position = None
                self.snake.positions.append(tail)
                self.score += 1
                reward = 1.0
                
            # Generate new food if there is none    
            if self.food.position is None:
                self.food.regenerate(np.where(self.pitch == 0))
            self.pitch[self.food.position] = 2
            
            # Update pitch
            self.update_pitch()
        else:
            reward = -100.0
        
        return self.pitch, reward, any(terminated), False, None
        
    def update_pitch(self, show:bool=False) -> None:
        self.pitch = np.zeros_like(self.pitch)
        
        # Update food and snake position on pitch
        self.pitch[self.food.position] = 2
        for pos in self.snake.positions:
            self.pitch[pos] = 1

        # Print pitch to console
        if show:
            pitch = np.full_like(self.pitch, " ", dtype=object)
            pitch[self.pitch == 1] = "S"    # Mark snake position with an "S"
            pitch[self.pitch == 2] = "F"    # Mark food position with an "F"
            print(pitch)
            

class Snake:
   
    def __init__(self, positions:tuple[int, int]) -> None:
        self.positions = positions
        self.direction = 2    # 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT 
        self.directions = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
        
    def move(self, bounds:tuple[int, int]) -> tuple[tuple[int, int], bool]:
        tail = self.positions[-1]
        terminated = False
        
        # Move each part of the "body" to the predecessor position
        for i in reversed(range(len(self.positions)-1)):
            self.positions[i+1] = self.positions[i]
            
        # Move the "head" in the current direction vector, only if inside pitch
        vector = self.directions[self.direction]
        new_position = tuple(sum(x) for x in zip(self.positions[0], vector))
        if (0 <= new_position[0] < bounds[0]) and (0 <= new_position[1] < bounds[1]):
            self.positions[0] = new_position
        else:
            terminated = True
            
        return tail, terminated
            
    def change_direction(self, new_direction:int) -> None:
        if new_direction == 1 and self.direction != 3:
            self.direction = new_direction
        if new_direction == 2 and self.direction != 4:
            self.direction = new_direction
        if new_direction == 3 and self.direction != 1:
            self.direction = new_direction
        if new_direction == 4 and self.direction != 2:
            self.direction = new_direction
            

class Food:
    
    def __init__(self) -> None:
        self.position = None

    def regenerate(self, positions) -> None:
        self.position = (np.random.choice(positions[0]), np.random.choice(positions[1]))



# Game Loop
# TODO: put this in the GAME class and make a render function 
if __name__ == "__main__":
    size = (40, 40)
    env = Game(size)
    terminated = False
    
    pygame.init()
    
    bg_color = (0, 0, 0)
    snake_color = (255, 255, 255)
    food_color = (0, 255, 0)
    score_color = (255, 0, 0)
    font = pygame.font.Font("freesansbold.ttf", 32)
    px = 15
    
    clock = pygame.time.Clock()
    
    screen = pygame.display.set_mode([s * px for s in size])
    pygame.display.set_caption("Snake")
    
    while True:
        action = 0
        score = font.render(str(env.score), True, score_color)
        game_over = font.render("GAME OVER", True, snake_color)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 2
                if event.key == pygame.K_DOWN:
                    action = 3
                if event.key == pygame.K_LEFT:
                    action = 4
        
        screen.fill(bg_color)
        screen.blit(score, (10, 10))
          
        if not terminated:
            _, _, terminated, _, _ = env.step(action)
        else:
            pygame.draw.rect(screen, score_color, (100, 200, 400, 200))
            screen.blit(game_over, (200, 280))
 
        # Draw snake
        for pos in env.snake.positions:
            pygame.draw.rect(screen, snake_color, (pos[1] * px, pos[0] * px, px, px))
        
        # Draw food
        pos = env.food.position
        pygame.draw.rect(screen, food_color, (pos[1] * px, pos[0] * px, px, px))
                
        clock.tick(15)        
        pygame.display.update()                 
 