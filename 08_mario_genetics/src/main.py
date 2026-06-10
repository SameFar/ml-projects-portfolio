from next_generation import mariojrs, natural_selection
from game_logic import create_map, goals, draw, move, clock
from pathlib import Path
import config
import numpy as np
from game_objects import Player
import math

import pygame



def main():
    SAVE_DIR = Path(__file__).resolve().parent.parent/"saved_model"/"elite_mario_dna.npy"
    create_map()
    agents = [Player() for _ in range(config.POPULATION)]

    generation = 1
    i = 0
    stagnant_gen = 0
    run = True
    headless = False
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    headless = not headless

                if event.key == pygame.K_s:
                    dna_pool = [agent.dna for agent in agents]

                    np.save(SAVE_DIR, np.array(dna_pool))
                    print(f"Generation {generation} saved")

                if event.key == pygame.K_l:
                    try:
                        loaded_dna = np.load(SAVE_DIR)
                        
                        # Rebuild a fresh list of Player objects using the loaded DNA
                        agents = [Player(dna=dna) for dna in loaded_dna]
                        
                        # Reset your simulation tracking metrics
                        i = 0
                        generation += 1 
                        print("Successfully loaded saved population to current generation")
                        
                    except FileNotFoundError:
                        print("404 file found. Press 'S' to save one first.")

        # Generation ends if the timer expires or everyone dies
        all_dead = all(not agent.alive for agent in agents)
        
        if i >= config.GENES or all_dead:
            # Get all metrics for fitness evaluation
            for agent in agents:
                dx, dy = agent.x, agent.y

                for goal in goals:
                    if (goal.x, goal.y) not in agent.checks:
                        distance_from_next_goal = math.sqrt((dx - goal.x)**2 + (dy - goal.y)**2)
                        break
                agent.distance = max(1, distance_from_next_goal)
                
                if len(agent.checks) == len(goals):
                    agent.bonus = 2000
            
            max_fitness = max([a.fitness() for a in agents])
            
            # Change mutation
            if int(config.PREV_FITNESS) >= int(max_fitness):
                stagnant_gen += 1
                if stagnant_gen >= 5 and stagnant_gen < 15:
                    config.MUTATION = 0.05

                elif stagnant_gen == 15:
                    run = False
            else:
                stagnant_gen = 0
                config.MUTATION = config.BASE_MUT

            print(f"Gen {generation} complete. Max Fitness: {max_fitness:.2f}")

            if generation == 150:
                run = False

            # Makes new generation of marios
            elites = natural_selection(agents)
            if not run:
                MARIO = elites[:0]
                with open('results.txt', 'w') as f:
                    f.write(f'''DNA: {MARIO.dna}
                    fitness: {MARIO.fitness()}''')
                    break
            agents = mariojrs(elites)
            
            generation += 1
            i = 0 # Reset frame tick
            config.PREV_FITNESS = max_fitness
            continue
        
        # Main (alive) loop
        for agent in agents:
            if not agent.alive:
                continue

            agent.time -= 1

            # Death criteria
            if agent.y > config.HEIGHT or agent.stagnant_count == 90 or agent.time == 0:
                agent.alive = 0
                continue

            current_goal_idx = min(len(agent.checks), len(goals) - 1)
            target_goal = goals[current_goal_idx]

            prev_dist_to_goal = abs(agent.x - target_goal.x)
            
            # Movement
            keys = agent.dna
            agent.speed = 0
            if keys[i][2] and not keys[i][1]:  # Right
                agent.speed = config.WALK_SPEED
            if keys[i][0] and not agent.jumping:  # Jump
                agent.velocity = config.VELOCITY
                agent.jumping = True
            if keys[i][1] and not keys[i][2]:  # Left
                agent.speed = -config.WALK_SPEED
            
            move(agent)

            dist_to_goal = abs(agent.x - target_goal.x)

            if abs(agent.x - agent.previous_x) < 40 or (prev_dist_to_goal - dist_to_goal) < 3:
                agent.stagnant_count += 1
            else:
                agent.previous_x = agent.x
                agent.stagnant_count = 0
        

        draw(headless, agents)
        pygame.display.update()
        clock.tick(60)  
        i += 1  # Increments once per parallel step

if __name__ == "__main__":
    main()