import random
import math
import torch
import torch.nn.functional as F
from config import device

def select_action(state, policy_net, n_actions, steps_done, eps_start, eps_end, eps_decay):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return None
    
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)

    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    reward_batch = torch.cat(batch_reward)
    done_batch = torch.tensor(batch_done, device=device, dtype=torch.float32)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    if len(non_final_next_states) > 0:
        with torch.no_grad():
            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
    
    expected_state_action_values = reward_batch + (gamma * next_state_values * (1 - done_batch))

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()
    
    return loss.item()