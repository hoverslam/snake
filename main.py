from snake import Game


if __name__ == "__main__":
    env = Game((40, 40)) 

    while True:        
        terminated = False
        env.reset()

        while not terminated:
            action = env.check_events()
            _, _, terminated, _, _ = env.step(action)
            env.render()
            
        print(f"Score: {env.score}")