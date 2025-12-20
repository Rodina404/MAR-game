import pygame
import random
import numpy as np


WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
BASKET_BROWN = (139, 69, 19)
GOOD_GREEN = (0, 200, 0)
BAD_DARK = (50, 50, 50)
STEM_BROWN = (101, 67, 33)
GROUND_GREEN = (34, 139, 34)


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
AGENT_WIDTH = 100
AGENT_HEIGHT = 20
AGENT_SPEED = 20
ITEM_SIZE = 30
ITEM_SPEED = 10
FPS = 30
MAX_STEPS_PER_EPISODE = 500


GRID_SIZE = 20
X_GRID_CELLS = SCREEN_WIDTH // GRID_SIZE
Y_GRID_CELLS = SCREEN_HEIGHT // GRID_SIZE


class CatchGameEnv:
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.MAX_STEPS = MAX_STEPS_PER_EPISODE
        
        self.action_space_size = 3
        
        self.observation_space_shape = (
            X_GRID_CELLS,
            X_GRID_CELLS,
            Y_GRID_CELLS,
            2
        )
        
        self.REWARD_CATCH_GOOD = 10
        self.REWARD_CATCH_BAD = -10
        self.REWARD_MISS_GOOD = -1
        self.REWARD_STEP = -0.01
        
        self.agent_x = 0
        self.item_x = 0
        self.item_y = 0
        self.item_type = 0
        self.item_active = False
        self.score = 0
        self.total_episodes = 0
        self.step_count = 0
        
    def _get_obs(self):
        agent_x_grid = self.agent_x // GRID_SIZE
        item_x_grid = self.item_x // GRID_SIZE
        item_y_grid = self.item_y // GRID_SIZE
        
        agent_x_grid = np.clip(agent_x_grid, 0, X_GRID_CELLS - 1)
        item_x_grid = np.clip(item_x_grid, 0, X_GRID_CELLS - 1)
        item_y_grid = np.clip(item_y_grid, 0, Y_GRID_CELLS - 1)
        
        return (int(agent_x_grid), int(item_x_grid), int(item_y_grid), int(self.item_type))


    def reset(self):
        self.total_episodes += 1
        
        self.agent_x = SCREEN_WIDTH // 2 - AGENT_WIDTH // 2
        
        self.score = 0 
        self.step_count = 0
        
        self._spawn_new_item()
        
        return self._get_obs(), {}


    def _spawn_new_item(self):
        self.item_x = random.randint(0, SCREEN_WIDTH - ITEM_SIZE)
        self.item_y = 0
        self.item_type = random.choice([0, 1])
        self.item_active = True


    def step(self, action):
        reward = self.REWARD_STEP
        terminated = False
        truncated = False 
        
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS:
            truncated = True
            
        if action == 0:
            self.agent_x -= AGENT_SPEED
        elif action == 2:
            self.agent_x += AGENT_SPEED
            
        self.agent_x = np.clip(self.agent_x, 0, SCREEN_WIDTH - AGENT_WIDTH)
        
        self.item_y += ITEM_SPEED
        
        if self.item_active:
            item_caught_or_missed = False
            
            agent_rect = pygame.Rect(self.agent_x, SCREEN_HEIGHT - AGENT_HEIGHT, AGENT_WIDTH, AGENT_HEIGHT)
            item_rect = pygame.Rect(self.item_x, self.item_y, ITEM_SIZE, ITEM_SIZE)
            
            if agent_rect.colliderect(item_rect):
                if self.item_type == 0:
                    reward += self.REWARD_CATCH_GOOD
                else:
                    reward += self.REWARD_CATCH_BAD
                
                item_caught_or_missed = True
                
            elif self.item_y > SCREEN_HEIGHT:
                if self.item_type == 0:
                    reward += self.REWARD_MISS_GOOD
                
                item_caught_or_missed = True
            
            if item_caught_or_missed:
                self.score += reward
                self.item_active = False
                self._spawn_new_item()
        
        observation = self._get_obs()
        info = {"score": self.score}
        
        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.font = pygame.font.Font(None, 36)
                    
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Catch Game RL Environment")
                self.clock = pygame.time.Clock()
                
            self.screen.fill(LIGHT_BLUE)


            ground_rect = pygame.Rect(0, SCREEN_HEIGHT - AGENT_HEIGHT, SCREEN_WIDTH, AGENT_HEIGHT)
            pygame.draw.rect(self.screen, GROUND_GREEN, ground_rect)
            
            agent_color = BASKET_BROWN
            agent_rect = pygame.Rect(self.agent_x, SCREEN_HEIGHT - AGENT_HEIGHT, AGENT_WIDTH, AGENT_HEIGHT)
            pygame.draw.rect(self.screen, agent_color, agent_rect)
            
            if self.item_active:
                item_color = GOOD_GREEN if self.item_type == 0 else BAD_DARK
                item_rect = pygame.Rect(self.item_x, self.item_y, ITEM_SIZE, ITEM_SIZE)
                
                if self.item_type == 0:
                    pygame.draw.ellipse(self.screen, item_color, item_rect)
                    pygame.draw.circle(self.screen, WHITE, (item_rect.x + ITEM_SIZE // 4, item_rect.y + ITEM_SIZE // 4), 3)
                    pygame.draw.line(self.screen, STEM_BROWN, (item_rect.centerx, item_rect.y), (item_rect.centerx + 3, item_rect.y - 5), 2)
                else:
                    pygame.draw.circle(self.screen, item_color, item_rect.center, ITEM_SIZE // 2)
                    pygame.draw.line(self.screen, STEM_BROWN, (item_rect.centerx, item_rect.y - 5), (item_rect.centerx + 5, item_rect.y - 10), 2)
                    pygame.draw.circle(self.screen, WHITE, (item_rect.centerx, item_rect.y - 5), 3)
            
            text = self.font.render(f"Score: {self.score} | Episode: {self.total_episodes} | Steps: {self.step_count}/{self.MAX_STEPS}", True, WHITE)
            self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            
            self.clock.tick(FPS)


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


if __name__ == '__main__':
    env = CatchGameEnv(render_mode='human')
    
    episodes = 3
    for episode in range(episodes):
        observation, info = env.reset()
        truncated = False
        
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        while not truncated:
            action = random.choice([0, 1, 2]) 
            
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            print(f"Step: {env.step_count}, Obs: {observation}, Reward: {reward:.2f}, Truncated: {truncated}, Score: {info['score']}")
            
            if terminated:
                print("WARNING: Episode terminated unexpectedly. Check logic.")
            
    env.close()
    print("\nSimulation finished.")
