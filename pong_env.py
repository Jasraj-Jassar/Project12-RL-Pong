import random
from dataclasses import dataclass

from config import GRID_HEIGHT, GRID_WIDTH, PADDLE_HEIGHT, PADDLE_X, SEED


@dataclass
class Ball:
    x: int
    y: int
    vx: int
    vy: int


class PongEnv:
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, paddle_height=PADDLE_HEIGHT, seed=SEED):
        self.width = width
        self.height = height
        self.paddle_height = paddle_height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.paddle_y = (self.height - self.paddle_height) // 2
        self.ball = Ball(
            x=self.width // 2,
            y=self.height // 2,
            vx=-1,
            vy=self.rng.choice([-1, 1]),
        )
        return self._get_state()

    def step(self, action):
        if action not in (-1, 0, 1):
            raise ValueError("Action must be -1, 0, or 1")

        self.paddle_y = max(0, min(self.height - self.paddle_height, self.paddle_y + action))

        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy

        if self.ball.y <= 0:
            self.ball.y = 0
            self.ball.vy *= -1
        elif self.ball.y >= self.height - 1:
            self.ball.y = self.height - 1
            self.ball.vy *= -1

        if self.ball.x >= self.width - 1:
            self.ball.x = self.width - 1
            self.ball.vx *= -1

        reward = 0
        done = False

        if self.ball.vx < 0 and self.ball.x == PADDLE_X:
            if self.paddle_y <= self.ball.y < self.paddle_y + self.paddle_height:
                self.ball.vx *= -1
                reward = 1
            else:
                reward = -1
                done = True
        elif self.ball.x < PADDLE_X:
            reward = -1
            done = True

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return (
            self.ball.x,
            self.ball.y,
            self.ball.vx,
            self.ball.vy,
            self.paddle_y,
        )

    def render(self):
        grid = [[" "] * self.width for _ in range(self.height)]

        for offset in range(self.paddle_height):
            y = self.paddle_y + offset
            if 0 <= y < self.height:
                grid[y][PADDLE_X] = "|"

        if 0 <= self.ball.y < self.height and 0 <= self.ball.x < self.width:
            grid[self.ball.y][self.ball.x] = "O"

        border = "+" + ("-" * self.width) + "+"
        lines = [border]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)
        return "\n".join(lines)
