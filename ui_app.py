import os
import tkinter as tk

from config import (
    CELL_SIZE,
    FRAME_DELAY_MS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAX_STEPS,
    PADDLE_HEIGHT,
    PADDLE_X,
    SEED,
)
from pong_env import PongEnv
from q_agent import QLearningAgent


class PongUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Beginner RL Pong")

        self.env = PongEnv(seed=SEED)
        self.agent = QLearningAgent(actions=[-1, 0, 1], seed=SEED)

        self.mode = "idle"
        self.running = False
        self.episode = 1
        self.step_in_episode = 0
        self.total_reward = 0
        self.hits = 0
        self.state = self.env.reset()
        self.q_path = "q_table.pkl"

        self.mode_var = tk.StringVar(value="mode: idle")
        self.episode_var = tk.StringVar(value="episode: 1")
        self.reward_var = tk.StringVar(value="reward: 0")
        self.hits_var = tk.StringVar(value="hits: 0")
        self.epsilon_var = tk.StringVar(value="epsilon: 1.000")
        self.message_var = tk.StringVar(value=f"q-table path: {self.q_path}")

        self._build_ui()
        self._update_labels()
        self._schedule_next_frame()

    def _build_ui(self):
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(padx=10, pady=6, fill="x")

        labels = [
            self.mode_var,
            self.episode_var,
            self.reward_var,
            self.hits_var,
            self.epsilon_var,
        ]
        for idx, var in enumerate(labels):
            tk.Label(stats_frame, textvariable=var, width=16, anchor="w").grid(
                row=0, column=idx, padx=4
            )

        canvas_width = GRID_WIDTH * CELL_SIZE
        canvas_height = GRID_HEIGHT * CELL_SIZE
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_width,
            height=canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#333",
        )
        self.canvas.pack(padx=10, pady=6)

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(padx=10, pady=4)

        tk.Button(controls_frame, text="Start Training", command=self.start_training).grid(
            row=0, column=0, padx=4, pady=2
        )
        tk.Button(controls_frame, text="Play (Greedy)", command=self.start_play).grid(
            row=0, column=1, padx=4, pady=2
        )
        tk.Button(controls_frame, text="Stop", command=self.stop).grid(
            row=0, column=2, padx=4, pady=2
        )
        tk.Button(controls_frame, text="Reset Episode", command=self.reset_episode).grid(
            row=0, column=3, padx=4, pady=2
        )
        tk.Button(controls_frame, text="New Agent", command=self.reset_agent).grid(
            row=1, column=0, padx=4, pady=2
        )
        tk.Button(controls_frame, text="Save Q-table", command=self.save_q).grid(
            row=1, column=1, padx=4, pady=2
        )
        tk.Button(controls_frame, text="Load Q-table", command=self.load_q).grid(
            row=1, column=2, padx=4, pady=2
        )

        message_label = tk.Label(self.root, textvariable=self.message_var, anchor="w")
        message_label.pack(padx=10, pady=(0, 8), fill="x")

    def _schedule_next_frame(self):
        self.root.after(FRAME_DELAY_MS, self.tick)

    def start_training(self):
        self.mode = "train"
        self.running = True
        self.message_var.set("training... watch the paddle learn to follow the ball")
        self._update_labels()

    def start_play(self):
        self.mode = "play"
        self.running = True
        self.message_var.set("playing with greedy actions")
        self._update_labels()

    def stop(self):
        self.running = False
        self.mode = "idle"
        self.message_var.set("stopped")
        self._update_labels()

    def reset_episode(self):
        self._start_new_episode()
        self.message_var.set("episode reset")
        self._update_labels()

    def reset_agent(self):
        self.agent = QLearningAgent(actions=[-1, 0, 1], seed=SEED)
        self.episode = 1
        self._start_new_episode()
        self.message_var.set("new agent created")
        self._update_labels()

    def save_q(self):
        self.agent.save(self.q_path)
        self.message_var.set(f"saved q-table to {self.q_path}")

    def load_q(self):
        if not os.path.exists(self.q_path):
            self.message_var.set(f"missing {self.q_path} (train first)")
            return
        self.agent.load(self.q_path)
        self.message_var.set(f"loaded q-table from {self.q_path}")

    def tick(self):
        if self.running:
            if self.mode == "train":
                action = self.agent.select_action(self.state)
            else:
                action = self.agent.best_action(self.state)

            next_state, reward, done, _ = self.env.step(action)

            if self.mode == "train":
                self.agent.update(self.state, action, reward, next_state, done)

            self.state = next_state
            self.total_reward += reward
            if reward > 0:
                self.hits += 1
            self.step_in_episode += 1

            if done or self.step_in_episode >= MAX_STEPS:
                self._finish_episode()

        self.draw()
        self._update_labels()
        self._schedule_next_frame()

    def _finish_episode(self):
        summary = (
            f"episode {self.episode} done reward={self.total_reward} "
            f"hits={self.hits}"
        )
        self.message_var.set(summary)
        if self.mode == "train":
            self.agent.decay_epsilon()
        self.episode += 1
        self._start_new_episode()

    def _start_new_episode(self):
        self.state = self.env.reset()
        self.step_in_episode = 0
        self.total_reward = 0
        self.hits = 0

    def _update_labels(self):
        self.mode_var.set(f"mode: {self.mode}")
        self.episode_var.set(f"episode: {self.episode}")
        self.reward_var.set(f"reward: {self.total_reward}")
        self.hits_var.set(f"hits: {self.hits}")
        self.epsilon_var.set(f"epsilon: {self.agent.epsilon:.3f}")

    def draw(self):
        self.canvas.delete("all")
        width = GRID_WIDTH * CELL_SIZE
        height = GRID_HEIGHT * CELL_SIZE
        self.canvas.create_rectangle(1, 1, width - 1, height - 1, outline="#333")

        paddle_x0 = PADDLE_X * CELL_SIZE
        paddle_y0 = self.env.paddle_y * CELL_SIZE
        paddle_x1 = (PADDLE_X + 1) * CELL_SIZE
        paddle_y1 = (self.env.paddle_y + PADDLE_HEIGHT) * CELL_SIZE
        self.canvas.create_rectangle(
            paddle_x0,
            paddle_y0,
            paddle_x1,
            paddle_y1,
            fill="#222",
            outline="#222",
        )

        ball_x0 = self.env.ball.x * CELL_SIZE
        ball_y0 = self.env.ball.y * CELL_SIZE
        ball_x1 = (self.env.ball.x + 1) * CELL_SIZE
        ball_y1 = (self.env.ball.y + 1) * CELL_SIZE
        self.canvas.create_oval(
            ball_x0,
            ball_y0,
            ball_x1,
            ball_y1,
            fill="#d64545",
            outline="#d64545",
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PongUI()
    app.run()
