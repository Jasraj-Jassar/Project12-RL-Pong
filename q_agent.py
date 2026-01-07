import pickle
import random


class QLearningAgent:
    def __init__(
        self,
        actions,
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
    ):
        self.actions = list(actions)
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q = {}

    def _get_qs(self, state):
        if state not in self.q:
            self.q[state] = [0.0 for _ in self.actions]
        return self.q[state]

    def best_action(self, state):
        qs = self._get_qs(state)
        max_q = max(qs)
        best_indices = [i for i, q in enumerate(qs) if q == max_q]
        return self.actions[self.rng.choice(best_indices)]

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(state)

    def update(self, state, action, reward, next_state, done):
        qs = self._get_qs(state)
        idx = self.action_to_index[action]
        if done:
            target = reward
        else:
            next_qs = self._get_qs(next_state)
            target = reward + self.gamma * max(next_qs)
        qs[idx] += self.alpha * (target - qs[idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.q, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.q = pickle.load(handle)
