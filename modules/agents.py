import random
import os
import torch
from torch import nn
from tqdm import trange


class DQNAgent(nn.Module):
    """A class representing an RL agent using a Deep Q-Network.
    """
    
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, learning_rate:float, 
                 gamma:float, epsilon:float, batch_size:int, memory_size:int=10000) -> None:
        """Initialize agent.

        Args:
            input_dim (int): Input dimension of the network.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Output dimension of the network (i.e. number of actions).
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for the Bellman equation [0, 1].
            epsilon (float): Initial exploration rate for the training [0, 1].
            batch_size (int): Batch size for the gradient descent.
            memory_size (int, optional): Size of the Replay Memory. Defaults to 10000.
        """
        super(DQNAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """A single forward pass through the network.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Raw output of the forward pass.
        """
        x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        logits = self.model(x)
        
        return logits
    
    def choose_action(self, x:torch.Tensor, epsilon:float=0.0) -> int:
        """Choose an action based on an observation.

        Args:
            x (torch.Tensor): An observation made by the agent.
            epsilon (float, optional): Probability to choose a random action. Defaults to 0.0.

        Returns:
            int: Index of action.
        """
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            logits = self(x)
            proba = nn.Softmax(dim=0)(logits)
            action = proba.argmax().item()
        
        return action
    
    def train(self, env, episodes:int, max_steps:int=10000) -> dict:
        """Training loop.

        Args:
            env (_type_): Environment on which the agent is trained.
            episodes (int): Number of training runs.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 10000.

        Returns:
            dict: Loss and score for each episode.
        """
        history = {"episode": [], "loss": [], "score": []}
        epsilon = self.epsilon
    
        pbar = trange(episodes)
        for e in pbar:
            terminated = False
            obs, _ = env.reset()
            episode_losses = []
            
            # Simple epsilon decay
            if e > episodes // 2:
                epsilon = max(epsilon * 0.99, 0.01)

            for _ in range(max_steps):
                env.check_events()
                action = self.choose_action(obs, epsilon)
                obs_, reward, terminated, _, _ = env.step(action)
                self.memory.add((obs, action, reward, obs_, terminated))        
                obs = obs_
                
                # Optimization
                batch = self.memory.sample(self.batch_size)
                batch[0].requires_grad_()
                batch[2].requires_grad_()
                batch[3].requires_grad_()
                batch[4].requires_grad_()
                loss = self.gradient_descent(batch)
                episode_losses.append(loss.item())

                if terminated:
                    break
            
            mean_loss = sum(episode_losses) / float(len(episode_losses))    
            pbar.set_description(f"loss={mean_loss:.6f}")
            history["episode"].append(e)
            history["loss"].append(mean_loss)
            history["score"].append(env.score)
            
        return history
    
    def gradient_descent(self, batch:list[torch.Tensor]) -> torch.Tensor:
        """An optimization step given a batch of experiences.

        Args:
            batch (list[torch.Tensor]): A batch of experiences.

        Returns:
            torch.Tensor: Average of all losses in this batch.
        """
        obs, action, reward, obs_, not_terminated = batch

        # Forward pass
        logits = self(obs)
        pred = logits.gather(1, action)
        target = reward + self.gamma * self(obs_).max(1, keepdim=True)[0] * not_terminated
        loss = ((target - pred)**2).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def save_model(self, file_name: str) -> None:
        """Save model.

        Args:
            file_name (str): File name of the model. A common PyTorch convention is 
            using .pt file extension. 
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, file_name: str) -> None:
        """Load model.
        
        Args:
            file_name (str): File name of the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class ReplayMemory:
    """Class that stores the agents experiences.
    """
    
    def __init__(self, size:int) -> None:
        """Initialize replay memory with a given size.

        Args:
            size (int): Size of the memory.
        """
        self.size = size
        self.data = []
        self.index = 0
        
    def add(self, experience:tuple) -> None:
        """Add experience to memory.

        Args:
            experience (tuple): Experience tuple (obs, action, reward, obs_, terminated).
        """
        if len(self.data) < self.size:
            self.data.append(experience)
        else:
            self.data[self.index] = experience

        self.index = (self.index + 1) % self.size
        
    def sample(self, batch_size:int) -> list[torch.Tensor]:
        """Sample a batch of experiences.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list[torch.Tensor]: A batch of experiences.
        """
        dim = self.data[0][0].shape[0]
        batch = [
            torch.empty((batch_size, dim), dtype=torch.float32),    # observations
            torch.empty((batch_size, 1), dtype=torch.long),         # actions
            torch.empty((batch_size, 1), dtype=torch.float32),      # rewards
            torch.empty((batch_size, dim), dtype=torch.float32),    # next observations
            torch.empty((batch_size, 1), dtype=torch.float32),      # not terminated
        ]
        
        sample = random.choices(self.data, k=batch_size)
        for i, (obs, action, reward, obs_, terminated) in enumerate(sample):
            batch[0][i,:] = torch.from_numpy(obs)
            batch[1][i] = action
            batch[2][i] = reward
            batch[3][i,:] = torch.from_numpy(obs_)
            batch[4][i] = not terminated
        
        return batch