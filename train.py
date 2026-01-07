import argparse

from config import MAX_STEPS, SEED
from pong_env import PongEnv
from q_agent import QLearningAgent


def train(episodes, save_path):
    env = PongEnv(seed=SEED)
    agent = QLearningAgent(actions=[-1, 0, 1], seed=SEED)

    episode_rewards = []
    episode_hits = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        hits = 0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            if reward > 0:
                hits += 1

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_hits.append(hits)

        if episode % 50 == 0:
            avg_reward = sum(episode_rewards[-50:]) / 50
            avg_hits = sum(episode_hits[-50:]) / 50
            print(
                f"episode={episode} avg_reward={avg_reward:.2f} avg_hits={avg_hits:.2f} epsilon={agent.epsilon:.3f}"
            )

    agent.save(save_path)
    print(f"saved q-table to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent to play Pong.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="q_table.pkl", help="Path to save the Q-table")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.episodes, args.save_path)
