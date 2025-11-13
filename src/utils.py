import cv2
import numpy as np
import torch
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from config import device

def get_screen(env):
    screen = env.render()
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_AREA)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).unsqueeze(0)

def update_plot(rewards, moving_avgs):
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    # Styling
    ax.plot(rewards, label='Reward', color='cyan', alpha=0.4)
    ax.plot(moving_avgs, label='Avg (50)', color='white', linewidth=2)
    ax.set_facecolor('#222222')
    fig.patch.set_facecolor('#222222')
    ax.tick_params(colors='white')
    
    ax.set_title("Training Performance", color='white')
    ax.set_xlabel("Episode", color='white')      
    ax.set_ylabel("Total Reward", color='white')  
    ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.2)
    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    plt.close(fig)
    
    return pygame.image.fromstring(raw_data, size, "RGB")
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.plot(rewards, label='Reward', color='cyan', alpha=0.4)
    ax.plot(moving_avgs, label='Avg (50)', color='white', linewidth=2)
    ax.set_facecolor('#222222')
    fig.patch.set_facecolor('#222222')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_title("Training Performance", color='white')
    ax.grid(True, linestyle='--', alpha=0.2)
    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    plt.close(fig)
    
    return pygame.image.fromstring(raw_data, size, "RGB")