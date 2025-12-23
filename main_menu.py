import pygame
import sys
import math
from q_learning import train_q_learning
from value_iteration import value_iteration
from policy_gradient import train_policy_gradient
from catch_env import CatchGameEnv
import numpy as np

# Modern color palette inspired by the game
SKY_BLUE = (135, 206, 250)
GRASS_GREEN = (76, 175, 80)
BASKET_BROWN = (121, 85, 72)
APPLE_RED = (244, 67, 54)
DARK_TEXT = (33, 33, 33)
WHITE = (255, 255, 255)
BUTTON_BLUE = (66, 165, 245)
BUTTON_GREEN = (102, 187, 106)
BUTTON_ORANGE = (255, 152, 0)
SHADOW = (0, 0, 0, 50)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
FPS = 60


class AnimatedButton:
    def __init__(self, x, y, width, height, text, color, icon_type, algorithm, delay):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.algorithm = algorithm
        self.icon_type = icon_type
        self.hover = False
        self.scale = 0
        self.target_scale = 1.0
        self.y_offset = 0
        self.delay = delay
        self.animation_started = False
        
    def start_animation(self, current_time):
        if current_time > self.delay:
            self.animation_started = True
    
    def draw_icon(self, surface, x, y, size):
        """Draw icon based on type"""
        if self.icon_type == "brain":
            # Brain icon (Q-Learning)
            pygame.draw.circle(surface, WHITE, (x, y), size)
            pygame.draw.circle(surface, self.color, (x, y), size, 3)
            pygame.draw.circle(surface, WHITE, (x - size//3, y - size//4), size//4)
            pygame.draw.circle(surface, WHITE, (x + size//3, y - size//4), size//4)
            
        elif self.icon_type == "diamond":
            # Diamond icon (Value Iteration)
            points = [
                (x, y - size),
                (x + size, y),
                (x, y + size),
                (x - size, y)
            ]
            pygame.draw.polygon(surface, WHITE, points)
            pygame.draw.polygon(surface, self.color, points, 3)
            
        elif self.icon_type == "rocket":
            # Rocket icon (Policy Gradient)
            pygame.draw.polygon(surface, WHITE, [
                (x, y - size),
                (x - size//2, y + size),
                (x + size//2, y + size)
            ])
            pygame.draw.circle(surface, self.color, (x, y), size//2)
    
    def draw(self, screen, font, time):
        if not self.animation_started:
            return
            
        # Smooth scale animation
        if abs(self.scale - self.target_scale) > 0.01:
            self.scale += (self.target_scale - self.scale) * 0.1
        
        # Hover bounce effect
        if self.hover:
            self.y_offset = math.sin(time * 0.1) * 3
            self.target_scale = 1.05
        else:
            self.y_offset = 0
            self.target_scale = 1.0
        
        # Calculate scaled dimensions
        scaled_width = int(self.rect.width * self.scale)
        scaled_height = int(self.rect.height * self.scale)
        scaled_rect = pygame.Rect(0, 0, scaled_width, scaled_height)
        scaled_rect.center = (self.rect.centerx, self.rect.centery + self.y_offset)
        
        # Draw shadow
        shadow_surface = pygame.Surface((scaled_width + 10, scaled_height + 10), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, SHADOW, shadow_surface.get_rect(), border_radius=15)
        screen.blit(shadow_surface, (scaled_rect.x - 5, scaled_rect.y + 5))
        
        # Draw button
        pygame.draw.rect(screen, self.color, scaled_rect, border_radius=15)
        
        # Draw white overlay on hover
        if self.hover:
            overlay = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (255, 255, 255, 30), overlay.get_rect(), border_radius=15)
            screen.blit(overlay, scaled_rect)
        
        # Draw icon
        icon_x = scaled_rect.x + 50
        icon_y = scaled_rect.centery
        self.draw_icon(screen, icon_x, icon_y, 20)
        
        # Draw text
        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(midleft=(icon_x + 50, scaled_rect.centery))
        screen.blit(text_surface, text_rect)
    
    def check_hover(self, mouse_pos):
        if not self.animation_started:
            self.hover = False
            return
        self.hover = self.rect.collidepoint(mouse_pos)
    
    def is_clicked(self, mouse_pos, mouse_clicked):
        if not self.animation_started:
            return False
        return self.rect.collidepoint(mouse_pos) and mouse_clicked


class FallingItem:
    def __init__(self):
        self.x = np.random.randint(50, SCREEN_WIDTH - 50)
        self.y = np.random.randint(-200, -50)
        self.speed = np.random.uniform(1, 3)
        self.item_type = np.random.choice(['apple', 'bomb'])
        self.rotation = 0
        self.rotation_speed = np.random.uniform(-2, 2)
        
    def update(self):
        self.y += self.speed
        self.rotation += self.rotation_speed
        if self.y > SCREEN_HEIGHT + 50:
            self.y = np.random.randint(-200, -50)
            self.x = np.random.randint(50, SCREEN_WIDTH - 50)
            self.item_type = np.random.choice(['apple', 'bomb'])
    
    def draw(self, screen):
        size = 25
        if self.item_type == 'apple':
            # Draw apple
            pygame.draw.circle(screen, APPLE_RED, (int(self.x), int(self.y)), size)
            pygame.draw.circle(screen, (255, 255, 255, 100), (int(self.x - 8), int(self.y - 8)), 6)
        else:
            # Draw bomb
            pygame.draw.circle(screen, (50, 50, 50), (int(self.x), int(self.y)), size)
            pygame.draw.line(screen, (100, 100, 100), 
                           (int(self.x), int(self.y - size)), 
                           (int(self.x + 5), int(self.y - size - 10)), 3)


class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Catch Game - Reinforcement Learning")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 80)
        self.subtitle_font = pygame.font.Font(None, 32)
        self.button_font = pygame.font.Font(None, 38)
        
        # Animation
        self.time = 0
        self.title_y = -100
        self.target_title_y = 100
        
        # Buttons
        button_width = 500
        button_height = 80
        button_x = (SCREEN_WIDTH - button_width) // 2
        start_y = 280
        spacing = 100
        
        self.buttons = [
            AnimatedButton(button_x, start_y, button_width, button_height,
                          "Q-Learning", BUTTON_BLUE, "brain", "q_learning", 20),
            AnimatedButton(button_x, start_y + spacing, button_width, button_height,
                          "Value Iteration", BUTTON_GREEN, "diamond", "value_iteration", 30),
            AnimatedButton(button_x, start_y + spacing * 2, button_width, button_height,
                          "Policy Gradient", BUTTON_ORANGE, "rocket", "policy_gradient", 40)
        ]
        
        # Falling items for background
        self.falling_items = [FallingItem() for _ in range(6)]
    
    def draw_basket(self, x, y):
        """Draw a small basket at position"""
        pygame.draw.rect(self.screen, BASKET_BROWN, (x - 30, y, 60, 15), border_radius=3)
        pygame.draw.line(self.screen, (80, 50, 30), (x - 25, y), (x + 25, y), 2)
    
    def run(self):
        running = True
        
        while running:
            self.time += 1
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_clicked = True
            
            # Animate title
            if abs(self.title_y - self.target_title_y) > 0.5:
                self.title_y += (self.target_title_y - self.title_y) * 0.1
            
            # Start button animations
            for button in self.buttons:
                button.start_animation(self.time)
                button.check_hover(mouse_pos)
                if button.is_clicked(mouse_pos, mouse_clicked):
                    # Close menu and run algorithm
                    pygame.quit()
                    self.run_algorithm(button.algorithm)
                    return  # Exit menu loop
            
            # Update falling items
            for item in self.falling_items:
                item.update()
            
            # Draw gradient background
            for y in range(SCREEN_HEIGHT):
                ratio = y / SCREEN_HEIGHT
                r = int(SKY_BLUE[0] + (GRASS_GREEN[0] - SKY_BLUE[0]) * ratio)
                g = int(SKY_BLUE[1] + (GRASS_GREEN[1] - SKY_BLUE[1]) * ratio)
                b = int(SKY_BLUE[2] + (GRASS_GREEN[2] - SKY_BLUE[2]) * ratio)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
            
            # Draw falling items
            for item in self.falling_items:
                item.draw(self.screen)
            
            # Draw title with shadow
            title_text = "Catch Game"
            shadow = self.title_font.render(title_text, True, (0, 0, 0, 100))
            shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH // 2 + 3, self.title_y + 3))
            self.screen.blit(shadow, shadow_rect)
            
            title = self.title_font.render(title_text, True, WHITE)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, self.title_y))
            self.screen.blit(title, title_rect)
            
            # Subtitle
            subtitle = self.subtitle_font.render("Reinforcement Learning", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, self.title_y + 60))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Draw decorative basket
            basket_bounce = math.sin(self.time * 0.05) * 5
            self.draw_basket(SCREEN_WIDTH // 2, 200 + basket_bounce)
            
            # Buttons
            for button in self.buttons:
                button.draw(self.screen, self.button_font, self.time)
            
            pygame.display.flip()
            self.clock.tick(FPS)
    
    def run_algorithm(self, algorithm):
        print(f"\n{'='*50}")
        print(f"Starting {algorithm.replace('_', ' ').title()}...")
        print(f"{'='*50}\n")
        
        if algorithm == "q_learning":
            train_q_learning()
            
        elif algorithm == "value_iteration":
            env = CatchGameEnv(render_mode=None)
            policy, V = value_iteration(env)
            np.save("value_iteration_policy.npy", policy)
            
        elif algorithm == "policy_gradient":
            train_policy_gradient()
        
        print(f"\n{'='*50}")
        print(f"✅ {algorithm.replace('_', ' ').title()} Completed!")
        print(f"{'='*50}\n")
    
    def demonstrate_policy(self, policy_file, algorithm_name):
        """Demonstrate the learned policy"""
        print(f"\n🎮 Demonstrating {algorithm_name} Policy...\n")
        
        env = CatchGameEnv(render_mode='human')
        
        # Load policy
        if "Q.npy" in policy_file:
            Q = np.load(policy_file, allow_pickle=True).item()
            policy_type = "q_learning"
        elif "policy.npy" in policy_file:
            policy = np.load(policy_file, allow_pickle=True).item()
            policy_type = "value_iteration"
        else:
            theta = np.load(policy_file, allow_pickle=True).item()
            policy_type = "policy_gradient"
        
        # Run demonstration episodes
        for episode in range(5):
            state, _ = env.reset()
            truncated = False
            
            while not truncated:
                # Choose action based on learned policy
                if policy_type == "q_learning":
                    if state in Q:
                        action = np.argmax(Q[state])
                    else:
                        action = 1  # Stay
                        
                elif policy_type == "value_iteration":
                    if state in policy:
                        action = policy[state]
                    else:
                        action = 1  # Stay
                        
                else:  # policy_gradient
                    if state in theta:
                        probs = self.softmax(theta[state])
                        action = np.argmax(probs)
                    else:
                        action = 1  # Stay
                
                state, reward, _, truncated, _ = env.step(action)
                env.render()
                
                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
        
        env.close()
        print(f"\n✅ {algorithm_name} demonstration completed!\n")
    
    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


if __name__ == "__main__":
    menu = MainMenu()
    menu.run()
