import pygame
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import ale_py
from config import *
from src.model import DQN
from src.memory import ReplayMemory
from src.agent import select_action, optimize_model
from src.utils import get_screen, update_plot

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
n_actions = env.action_space.n

policy_net = DQN(84, 84, n_actions).to(device)
target_net = DQN(84, 84, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)
steps_done = 0

pygame.init()
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 40, bold=True)
screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("Modular DQN Breakout")
clock = pygame.time.Clock()

episode_rewards = []
moving_averages = []
graph_surface = None

for i_episode in range(NUM_EPISODES):
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen
    total_reward = 0
    current_epsilon = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action_tensor, current_epsilon = select_action(
            state, policy_net, n_actions, steps_done, EPS_START, EPS_END, EPS_DECAY
        )
        steps_done += 1
        action = action_tensor.item()
        
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = get_screen(env)
        
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action_tensor, next_state, reward, done)
        state = next_state

        optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA)
        
        screen.fill((30, 30, 30))
        
        raw_frame = env.render()
        frame_surface = pygame.surfarray.make_surface(np.transpose(raw_frame, (1, 0, 2)))
        frame_surface = pygame.transform.scale(frame_surface, (GAME_WIDTH, GAME_HEIGHT))
        screen.blit(frame_surface, (0, 0))
        
        pygame.draw.line(screen, (100, 100, 100), (GAME_WIDTH, 0), (GAME_WIDTH, TOTAL_HEIGHT), 3)
        
        text_x = GAME_WIDTH + 30
        y_offset = 40
        
        screen.blit(large_font.render(f"Episode: {i_episode}", True, (255, 255, 255)), (text_x, y_offset))
        y_offset += 60
        
        screen.blit(font.render(f"Score: {total_reward:.0f}", True, (0, 255, 0)), (text_x, y_offset))
        y_offset += 40
        
        action_str = ACTION_NAMES.get(action, "UNKNOWN")
        color = (255, 0, 0) if action == 1 else (0, 255, 255)
        screen.blit(font.render(f"Action: {action_str}", True, color), (text_x, y_offset))
        y_offset += 40

        screen.blit(font.render(f"Epsilon: {current_epsilon:.4f}", True, (200, 200, 0)), (text_x, y_offset))
        y_offset += 50

        if graph_surface:
            screen.blit(graph_surface, (text_x - 10, y_offset))
        
        pygame.display.flip()
        clock.tick(FPS)

        if done:
            episode_rewards.append(total_reward)
            avg = np.mean(episode_rewards[-50:])
            moving_averages.append(avg)
            graph_surface = update_plot(episode_rewards, moving_averages)
            print(f"Episode {i_episode} finished. Reward: {total_reward}, Avg: {avg:.2f}")
            break
            
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

pygame.quit()