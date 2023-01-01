from modules.snake import Game
from modules.agents import DQNAgent
from tqdm import trange
import matplotlib.pyplot as plt


env = Game()
agent = DQNAgent(33, 512, 4, 1e-4, 0.99, 0.3, 64)
       
def train_agent(episodes):
    history = agent.train(env, episodes, 10000)
    agent.save_model("dqn.pt")
    plt.scatter(x=history["episode"], y=history["score"], s=3, alpha=0.8)
    plt.show()
    
def evaluate_agent(episodes):
    agent.load_model("dqn.pt")
    agent.model.eval()
    scores = []
    
    pbar = trange(episodes)
    for _ in pbar:
        obs, _ = env.reset()
        terminated = False
  
        for _ in range(10000):
            env.check_events()
            action = agent.choose_action(obs)
            obs, _, terminated, _, _ = env.step(action)
            
            if terminated:
                break
                    
        pbar.set_description(f"score={env.score}")
        scores.append(env.score)
    
    mean_score = sum(scores) / float(len(scores))    
    print(f"Average score over {episodes} episodes: {mean_score:.2f}")
        
def play(player="human"):
    if player == "ai":
        agent.load_model("dqn.pt")
        agent.model.eval()
        fps = 50
    else:
        fps=15
    
    while True:
        obs, _ = env.reset()
        terminated = False
  
        while not terminated:
            action = env.check_events()
            if player == "ai":
                action = agent.choose_action(obs)
            obs, _, terminated, _, _ = env.step(action)
            
            env.render(fps)
            
        print(f"Score: {env.score}")


if __name__ == "__main__":
    #train_agent(1000)
    #evaluate_agent(1000)
    
    play("ai")