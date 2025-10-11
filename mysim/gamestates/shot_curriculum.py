from collections import deque
from rlgym_sim.utils.gamestates import GameState

class CurriculumManager:
    def __init__(self, difficulty_levels=5):
        self.current_level = 0
        self.max_level = difficulty_levels - 1
        self.episodes_per_level = 1000
        self.episode_counter = 0
        self.success_threshold = 0.6  # 60% success rate to advance
        self.recent_successes = deque(maxlen=100)
    
    def update_curriculum(self, episode_success):
        # Track success (whether a goal was scored)
        self.recent_successes.append(float(episode_success))
        self.episode_counter += 1
        
        # Check if we should advance to the next level
        if (self.episode_counter >= self.episodes_per_level and 
            len(self.recent_successes) >= 50 and
            sum(self.recent_successes) / len(self.recent_successes) >= self.success_threshold and
            self.current_level < self.max_level):
            
            self.current_level += 1
            self.episode_counter = 0
            print(f"Advancing to curriculum level {self.current_level}")
    
    def get_spawn_config(self):
        # Level 0: Random positions in opponent half, empty goal
        # Level 1: Add restricted scoring areas (blockers)
        # Level 2: Add stationary defender
        # Level 3: Add slow-moving defender
        # Level 4: Full defender AI
        
        spawn_config = {
            "ball_position_range": None,
            "car_position_range": None,
            "opponent_difficulty": None,
            "spawn_opponents": False,
            "game_state": GameState()
        }
        
        # Configure based on current level
        if self.current_level == 0:
            spawn_config["ball_position_range"] = [(0, 2000), (-3000, 3000), (100, 500)]
            spawn_config["car_position_range"] = [(0, 1500), (-2000, 2000), (17, 17)]
            spawn_config["opponent_difficulty"] = 0  # No opponents
            spawn_config["spawn_opponents"] = False
        elif self.current_level == 1:
            # Similar to level 0 but add goal blockers
            spawn_config["ball_position_range"] = [(0, 2500), (-3000, 3000), (100, 500)]
            spawn_config["car_position_range"] = [(0, 2000), (-2000, 2000), (17, 17)]
            spawn_config["opponent_difficulty"] = 0
            spawn_config["spawn_opponents"] = False
        elif self.current_level == 2:
            # Add stationary defender
            spawn_config["ball_position_range"] = [(0, 2500), (-3000, 3000), (100, 500)]
            spawn_config["car_position_range"] = [(0, 2000), (-2000, 2000), (17, 17)]
            spawn_config["opponent_difficulty"] = 1  # Stationary defender
            spawn_config["spawn_opponents"] = True
        elif self.current_level == 3:
            # Add slow-moving defender
            spawn_config["ball_position_range"] = [(0, 2500), (-3000, 3000), (100, 500)]
            spawn_config["car_position_range"] = [(0, 2000), (-2000, 2000), (17, 17)]
            spawn_config["opponent_difficulty"] = 2  # Slow-moving defender
            spawn_config["spawn_opponents"] = True
        elif self.current_level == 4:
            # Full defender AI
            spawn_config["ball_position_range"] = [(0, 2500), (-3000, 3000), (100, 500)]
            spawn_config["car_position_range"] = [(0, 2000), (-2000, 2000), (17, 17)]
            spawn_config["opponent_difficulty"] = 3  # Full defender AI
            spawn_config["spawn_opponents"] = True
        
        return spawn_config

    def get_current_level(self):
        return self.current_level