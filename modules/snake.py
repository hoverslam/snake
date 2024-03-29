import numpy as np
import pygame
import sys


class Game:
    """A class representing the game logic.
    """

    def __init__(self, size: tuple[int, int] = (40, 40)) -> None:
        """Initialize the game.

        Args:
            size (tuple[int, int], optional): Size of the playing field. Defaults to (40, 40).
        """
        self.size = size

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset game to initial state.

        Returns:
            np.ndarray: Initial observation and the playing field
        """
        self.pitch = np.zeros(self.size, dtype=int)
        self.score = 0
        self.terminated = False

        # Initialize snake with a centered starting position and a length of 3
        self.snake = Snake([(self.size[0] // 2, self.size[1] // 2 - i) for i in range(3)])

        # Initialize food with a random starting position
        self.food = Food((np.random.choice(self.size[0]), np.random.choice(self.size[1])))

        self.update_pitch()

        # Initialize screen for rendering
        pygame.init()
        pygame.display.set_caption("Snake")
        self.font = pygame.font.Font("freesansbold.ttf", 20)
        self.px = 15    # Size of a "tile" in pixels
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode([s * self.px for s in (self.size[1], self.size[0])])

        return self.create_observation(), self.pitch

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, np.ndarray]:
        """Make one step given an action.

        Args:
            action (int): Action to take this step. (0: no action, 1: up, 2: right, 3: down, 4: left)

        Returns:
            tuple[np.ndarray, float, bool, bool, np.ndarray]: observation, reward, terminated, truncated, info
        """
        reward = 0.0

        # Change direction and move snake
        self.snake.change_direction(action)
        self.terminated = self.snake.move(self.size)

        # Only move on if the game is not terminated
        if not self.terminated:
            # If snake has eaten the food increase score and body length
            if self.snake.positions[0] == self.food.position:
                self.food.position = (-1, -1)
                self.snake.grow()
                self.score += 1
                reward = 1.0

            # Generate new food if there is none
            if self.food.position is (-1, -1):
                self.food.generate(np.argwhere(self.pitch == 0))

        else:
            reward = -1.0

        # Update pitch
        self.update_pitch()

        return self.create_observation(), reward, self.terminated, False, self.pitch

    def update_pitch(self) -> None:
        """Update the pitch.
        """
        self.pitch = np.zeros_like(self.pitch)

        # Update food position
        self.pitch[self.food.position] = 2

        # Update snake position
        for pos in self.snake.positions:
            self.pitch[pos] = 1

    def render(self, fps: int = 15) -> None:
        """Render game to screen.

        Args:
            fps (int, optional): Frames per second. Defaults to 15.
        """
        # Draw background
        self.screen.fill((0, 0, 0))

        # Draw score
        txt_score = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(txt_score, (10, 10))

        # Draw snake
        border_color = (0, 150, 0)
        for i, pos in enumerate(self.snake.positions):
            if i == 0:
                body_color = border_color
            else:
                body_color = (0, 200, 0)
            pygame.draw.rect(self.screen, body_color,
                             (pos[1] * self.px, pos[0] * self.px, self.px, self.px))
            pygame.draw.rect(self.screen, border_color,
                             (pos[1] * self.px, pos[0] * self.px, self.px, self.px),
                             width=2)

        # Draw food
        pos = self.food.position
        pygame.draw.rect(self.screen, (200, 0, 0),
                         (pos[1] * self.px, pos[0] * self.px, self.px, self.px),
                         border_radius=(self.px // 2))

        # Update screen
        self.clock.tick(fps)
        pygame.display.update()

    def check_events(self) -> int:
        """Check for keyboard strokes.

        Returns:
            int: Action corresponding the keyboard stroke.
        """
        action = 9    # no action, placeholder if no key is pushed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_RIGHT:
                    action = 1
                if event.key == pygame.K_DOWN:
                    action = 2
                if event.key == pygame.K_LEFT:
                    action = 3
                if event.key == pygame.K_r:
                    self.reset()

        return action

    def create_observation(self, size: int = 5) -> np.ndarray:
        """Creates an observation containing the near vicinity of the head, the snakes current 
        direction, and the direction to the food.

        Args:
            size (int, optional): The size of the vicinity the snake "sees". A size x size grid. Defaults to 5.

        Returns:
            np.ndarray: An observation of the current state.
        """
        head = self.snake.positions[0]

        # Make an array showing the vicinity (5x5) of the snakes head with ones indicating danger.
        vicinity = np.zeros((size, size), dtype=int)
        min = int(-np.floor(size / 2))
        max = int(np.ceil(size / 2))
        for i, x_offset in enumerate(range(min, max)):
            for j, y_offset in enumerate(range(min, max)):
                pos = (head[0] + x_offset, head[1] + y_offset)
                if (0 <= pos[0] < self.size[0]) and (0 <= pos[1] < self.size[1]):
                    vicinity[i, j] = self.pitch[pos[0], pos[1]]
                else:
                    vicinity[i, j] = 1

        # If food is in the vicinity set it to zero since this is not a threat.
        vicinity[np.where(vicinity == 2)] = 0

        # One-hot encode the current direction
        snake_directions = np.zeros(4, dtype=int)
        snake_directions[self.snake.direction] = 1

        # Find the direction in which the food is located.
        food_directions = np.zeros(4, dtype=int)
        vec = (head[0] - self.food.position[0], head[1] - self.food.position[1])
        food_directions[0] = 1 if vec[0] > 0 else 0
        food_directions[1] = 1 if vec[1] < 0 else 0
        food_directions[2] = 1 if vec[0] < 0 else 0
        food_directions[3] = 1 if vec[1] > 0 else 0

        # Combine vicinity, snake_directions and food_directions to one observation vector.
        observation = np.concatenate([vicinity.flatten(), snake_directions, food_directions])

        return observation.astype(np.float32)


class Snake:
    """A class representing a snake moving over the pitch.
    """

    def __init__(self, positions: list[tuple[int, int]]) -> None:
        """Initialize the class.

        Args:
            positions (list[tuple[int, int]]): The snakes starting position.
        """
        self.positions = positions
        self.direction = 1    # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def move(self, bounds: tuple[int, int]) -> bool:
        """Move the snake according to its current direction.

        Args:
            bounds (tuple[int, int]): The bounds of the pitch (i.e. its size). 

        Returns:
            bool: Flag indicated death of snake.
        """
        self.tail = self.positions[-1]    # Keep a "ghost tail" to easily increase body
        dead = False

        # Move each part of the "body" to the predecessor position
        for i in reversed(range(len(self.positions)-1)):
            self.positions[i+1] = self.positions[i]

        # Move the "head" in the current direction vector, only if inside pitch
        vector = self.directions[self.direction]
        new_position = tuple(sum(x) for x in zip(self.positions[0], vector))
        if (0 <= new_position[0] < bounds[0]) and (0 <= new_position[1] < bounds[1]):
            self.positions[0] = new_position
        else:
            dead = True

        if not dead and self.positions[0] in self.positions[1:]:
            dead = True

        return dead

    def change_direction(self, action: int) -> None:
        """Check whether the new direction is allowed given an action.  

        Args:
            action (int): An action.
        """
        if action == 0 and self.direction != 2:
            self.direction = action
        if action == 1 and self.direction != 3:
            self.direction = action
        if action == 2 and self.direction != 0:
            self.direction = action
        if action == 3 and self.direction != 1:
            self.direction = action

    def grow(self) -> None:
        """Grow the snake by one tile.
        """
        self.positions.append(self.tail)


class Food:
    """Class representing the food for the snake.
    """

    def __init__(self, position: tuple[int, int]) -> None:
        """Initialize class.

        Args:
            position (tuple[int, int]): The initial position.
        """
        self.position = position

    def generate(self, positions: np.ndarray) -> None:
        """Generate a new food source at a random position.

        Args:
            positions (np.ndarray): Possible positions for food.
        """
        idx = np.arange(positions.shape[0])
        self.position = tuple(positions[np.random.choice(idx)])
