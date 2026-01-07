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
    SMOOTHING,
)
from pong_env import PongEnv
from q_agent import QLearningAgent

BG = "#0f111a"
PANEL_BG = "#151823"
CANVAS_BG = "#0b0d14"
GRID_COLOR = "#1b1f2a"
BORDER_COLOR = "#2a3040"
TEXT = "#e6e8ef"
MUTED = "#9aa3b2"
ACCENT = "#66e3c4"
ACCENT_ALT = "#7aa2f7"
BALL_COLOR = "#ff6b6b"
BALL_GLOW = "#ff9c9c"
PADDLE_COLOR = "#e6e8ef"

TITLE_FONT = ("Fira Sans", 20, "bold")
LABEL_FONT = ("Fira Sans", 12)
STAT_FONT = ("Fira Code", 11)
BUTTON_FONT = ("Fira Sans", 11, "bold")


class PongUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RL Pong Trainer")
        self.root.configure(bg=BG)

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

        self.display_paddle_y = float(self.env.paddle_y)
        self.display_ball_x = float(self.env.ball.x)
        self.display_ball_y = float(self.env.ball.y)

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
        title = tk.Label(
            self.root,
            text="RL Pong Trainer",
            bg=BG,
            fg=TEXT,
            font=TITLE_FONT,
        )
        title.pack(padx=20, pady=(18, 4), anchor="w")

        subtitle = tk.Label(
            self.root,
            text="Watch a Q-learning agent learn to track the ball",
            bg=BG,
            fg=MUTED,
            font=LABEL_FONT,
        )
        subtitle.pack(padx=20, pady=(0, 12), anchor="w")

        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(padx=20, pady=(0, 18))

        left_frame = tk.Frame(main_frame, bg=BG)
        left_frame.grid(row=0, column=0, sticky="n")

        canvas_width = GRID_WIDTH * CELL_SIZE
        canvas_height = GRID_HEIGHT * CELL_SIZE
        self.canvas = tk.Canvas(
            left_frame,
            width=canvas_width,
            height=canvas_height,
            bg=CANVAS_BG,
            highlightthickness=2,
            highlightbackground=BORDER_COLOR,
        )
        self.canvas.pack()

        right_frame = tk.Frame(
            main_frame,
            bg=PANEL_BG,
            highlightbackground=BORDER_COLOR,
            highlightthickness=2,
        )
        right_frame.grid(row=0, column=1, sticky="n", padx=(16, 0))

        stats_title = tk.Label(
            right_frame,
            text="STATS",
            bg=PANEL_BG,
            fg=MUTED,
            font=LABEL_FONT,
        )
        stats_title.pack(padx=14, pady=(14, 8), anchor="w")

        stats_frame = tk.Frame(right_frame, bg=PANEL_BG)
        stats_frame.pack(padx=14, pady=(0, 12), fill="x")

        labels = [
            self.mode_var,
            self.episode_var,
            self.reward_var,
            self.hits_var,
            self.epsilon_var,
        ]
        for var in labels:
            tk.Label(
                stats_frame,
                textvariable=var,
                bg=PANEL_BG,
                fg=TEXT,
                font=STAT_FONT,
                anchor="w",
            ).pack(fill="x", pady=2)

        controls_title = tk.Label(
            right_frame,
            text="CONTROLS",
            bg=PANEL_BG,
            fg=MUTED,
            font=LABEL_FONT,
        )
        controls_title.pack(padx=14, pady=(6, 8), anchor="w")

        controls_frame = tk.Frame(right_frame, bg=PANEL_BG)
        controls_frame.pack(padx=14, pady=(0, 12), fill="x")

        self._add_button(
            controls_frame,
            "Start Training",
            self.start_training,
            bg=ACCENT,
            fg=BG,
        )
        self._add_button(
            controls_frame,
            "Play (Greedy)",
            self.start_play,
            bg=ACCENT_ALT,
            fg=BG,
        )
        self._add_button(
            controls_frame,
            "Stop",
            self.stop,
            bg="#2d3347",
            fg=TEXT,
        )
        self._add_button(
            controls_frame,
            "Reset Episode",
            self.reset_episode,
            bg="#2d3347",
            fg=TEXT,
        )
        self._add_button(
            controls_frame,
            "New Agent",
            self.reset_agent,
            bg="#2d3347",
            fg=TEXT,
        )
        self._add_button(
            controls_frame,
            "Save Q-table",
            self.save_q,
            bg="#2d3347",
            fg=TEXT,
        )
        self._add_button(
            controls_frame,
            "Load Q-table",
            self.load_q,
            bg="#2d3347",
            fg=TEXT,
        )

        message_label = tk.Label(
            right_frame,
            textvariable=self.message_var,
            bg=PANEL_BG,
            fg=MUTED,
            font=STAT_FONT,
            anchor="w",
            justify="left",
            wraplength=240,
        )
        message_label.pack(padx=14, pady=(4, 14), fill="x")

    def _add_button(self, parent, text, command, bg, fg):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            font=BUTTON_FONT,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=6,
            cursor="hand2",
        )
        button.pack(fill="x", pady=4)

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

        self._update_display_state()
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
        self.display_paddle_y = float(self.env.paddle_y)
        self.display_ball_x = float(self.env.ball.x)
        self.display_ball_y = float(self.env.ball.y)

    def _update_display_state(self):
        self.display_paddle_y = self._smooth(self.display_paddle_y, self.env.paddle_y)
        self.display_ball_x = self._smooth(self.display_ball_x, self.env.ball.x)
        self.display_ball_y = self._smooth(self.display_ball_y, self.env.ball.y)

    @staticmethod
    def _smooth(current, target):
        return current + (target - current) * SMOOTHING

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
        self.canvas.create_rectangle(
            0,
            0,
            width,
            height,
            outline=BORDER_COLOR,
            fill=CANVAS_BG,
            width=2,
        )

        for x in range(1, GRID_WIDTH):
            x_pos = x * CELL_SIZE
            self.canvas.create_line(x_pos, 0, x_pos, height, fill=GRID_COLOR)
        for y in range(1, GRID_HEIGHT):
            y_pos = y * CELL_SIZE
            self.canvas.create_line(0, y_pos, width, y_pos, fill=GRID_COLOR)

        paddle_x0 = PADDLE_X * CELL_SIZE + 2
        paddle_y0 = self.display_paddle_y * CELL_SIZE + 2
        paddle_x1 = (PADDLE_X + 1) * CELL_SIZE - 2
        paddle_y1 = (self.display_paddle_y + PADDLE_HEIGHT) * CELL_SIZE - 2
        self.canvas.create_rectangle(
            paddle_x0,
            paddle_y0,
            paddle_x1,
            paddle_y1,
            fill=PADDLE_COLOR,
            outline=PADDLE_COLOR,
        )

        ball_pad = CELL_SIZE * 0.18
        ball_x0 = self.display_ball_x * CELL_SIZE + ball_pad
        ball_y0 = self.display_ball_y * CELL_SIZE + ball_pad
        ball_x1 = (self.display_ball_x + 1) * CELL_SIZE - ball_pad
        ball_y1 = (self.display_ball_y + 1) * CELL_SIZE - ball_pad
        glow_pad = ball_pad * 0.5
        self.canvas.create_oval(
            ball_x0 - glow_pad,
            ball_y0 - glow_pad,
            ball_x1 + glow_pad,
            ball_y1 + glow_pad,
            fill=BALL_GLOW,
            outline=BALL_GLOW,
        )
        self.canvas.create_oval(
            ball_x0,
            ball_y0,
            ball_x1,
            ball_y1,
            fill=BALL_COLOR,
            outline=BALL_COLOR,
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PongUI()
    app.run()
