from snake import Game


if __name__ == "__main__":
    size = (40, 40)
    env = Game(size)
    terminated = False
    
    while True:
        action = env.check_events()
        
        if action == 9:
            env.restart()
            terminated = False 
        
        if not terminated:
            _, _, terminated, _, _ = env.step(action)

        env.render()