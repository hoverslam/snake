import numpy as np
import pygame
import sys


class Game:
    
    def __init__(self, size:tuple[int, int]=(40, 40)) -> None:
        self.size = size
        self.restart()
        
    def restart(self):
        self.pitch = np.zeros(self.size, dtype=int)
        self.score = 0
        self.terminated = [False, False]
        
        # Initialize snake with a centered starting position and a length of 3
        self.snake = Snake([(self.size[0] // 2, self.size[1] // 2 - i) for i in range(3)])
        
        # Initialize food
        self.food = Food()
        
        # Initialize screen for rendering
        pygame.init()
        pygame.display.set_caption("Snake")    
        self.bg_color = (0, 0, 0)
        self.snake_color = (255, 255, 255)
        self.food_color = (0, 255, 0)
        self.score_color = (255, 0, 0)
        self.font = pygame.font.Font("freesansbold.ttf", 32)
        self.px = 15        
        self.clock = pygame.time.Clock()        
        self.screen = pygame.display.set_mode([s * self.px for s in self.size])
      
    def step(self, action:int) -> tuple[np.ndarray, float, bool, bool, None]:
        reward = 0.0
        
        # Change direction if action is not zero
        if action != 0:
            self.snake.change_direction(action)
               
        # Move snake
        tail, self.terminated[0] = self.snake.move(self.size)

        # If snake has bitten its own body the game is over
        self.terminated[1] = self.snake.positions[0] in self.snake.positions[1:]
        
        # Only move on if the game is not terminated
        if not any(self.terminated):        
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
        
        return self.pitch, reward, any(self.terminated), False, None
        
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
            
    def render(self) -> None:
        self.screen.fill(self.bg_color)
        
        # Draw score
        txt_score = self.font.render(str(self.score), True, self.score_color)
        self.screen.blit(txt_score, (10, 10))
        
        # Draw snake
        for pos in self.snake.positions:
            pygame.draw.rect(self.screen, self.snake_color, 
                             (pos[1] * self.px, pos[0] * self.px, self.px, self.px))
        
        # Draw food
        pos = self.food.position
        pygame.draw.rect(self.screen, self.food_color, 
                         (pos[1] * self.px, pos[0] * self.px, self.px, self.px))
        
        # 
        if any(self.terminated):
            pygame.draw.rect(self.screen, self.score_color, (100, 200, 400, 200))
            txt_game_over = self.font.render("GAME OVER", True, self.snake_color)
            self.screen.blit(txt_game_over, (200, 260))
            txt_restart = self.font.render("Press ENTER to restart", True, self.snake_color)
            self.screen.blit(txt_restart, (120, 310))
            
        # Update screen
        self.clock.tick(15)        
        pygame.display.update()
    
    def check_events(self) -> int:
        action = 0
        
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
                if event.key == pygame.K_RETURN:
                    action = 9
        
        return action
            

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
        self.position = (np.random.choice(positions[1]), np.random.choice(positions[0]))