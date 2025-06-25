from agentes.base import Agent
import numpy as np
import random
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table_final.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Acceder directamente a las propiedades del juego
        self.game_pipe_gap = self.game.pipe_gap
        self.game_height = self.game.height 
        self.game_width = self.game.width

        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # Parámetros de discretización
        self.num_bins = {
            'player_y': 10,   
            'delta_y': 10,
            'delta_x': 10
        }
        self.player_v_threshold = 0
        self.player_vel_threshold_slow = 2 
        self.player_vel_threshold_fast = 5
        self.player_x_threshold = 0.5

    def discretize_state(self, state_dict):
        # 1. Posicion en Y discretizada
        player_y = state_dict['player_y']
        
        player_norm_y = player_y / self.game_height
        player_y_bin = int(np.clip(player_norm_y * self.num_bins['player_y'], 0, self.num_bins['player_y'] - 1))

        # 2. Discretizar la velocidad del jugador
        pvy = state_dict['player_vel']
        if pvy < -self.player_vel_threshold_fast:   # Arriba muy rápido
            player_velocity_bin = 0
        elif pvy < -self.player_vel_threshold_slow: # Arriba rápido
            player_velocity_bin = 1
        elif pvy <= self.player_vel_threshold_slow: # Lento o quieto
            player_velocity_bin = 2
        elif pvy <= self.player_vel_threshold_fast: # Abajo rápido
            player_velocity_bin = 3
        else:                                       # Abajo muy rápido
            player_velocity_bin = 4
        
        # 3. Bin a partir de la diferencia vertical entre el player y el centro del gap
        center_gap = state_dict['next_pipe_bottom_y'] - self.game_pipe_gap / 2
        delta_y = center_gap - player_y

        delta_y_norm = (delta_y + self.game_height) / (2 * self.game_height)
        delta_y_bin = int(np.clip(delta_y_norm * self.num_bins['delta_y'], 0, self.num_bins['delta_y'] - 1))

        # 4. Diferencia vertical entre el player y bottom
        delta_bottom_y = state_dict['next_pipe_bottom_y'] - player_y
        delta_bottom_y_norm = (delta_bottom_y + self.game_height) / (2 * self.game_height)
        delta_bottom_y_bin = int(np.clip(delta_bottom_y_norm * self.num_bins['delta_y'], 0, self.num_bins['delta_y'] - 1))

        # 5. Diferencia vertical entre el player y top
        delta_top_y = state_dict['next_pipe_top_y'] - player_y
        delta_top_y_norm = (delta_top_y + self.game_height) / (2 * self.game_height)
        delta_top_y_bin = int(np.clip(delta_top_y_norm * self.num_bins['delta_y'], 0, self.num_bins['delta_y'] - 1))

        # 6. Discretizar distancia entre el player y la pipe
        dist_x = state_dict['next_pipe_dist_to_player'] / self.game_width
        dist_x_bin = int(np.clip(dist_x * self.num_bins['delta_x'], 0, self.num_bins['delta_x'] - 1))

        return (
            player_y_bin,
            player_velocity_bin,
            delta_y_bin,
            delta_bottom_y_bin,
            delta_top_y_bin,
            dist_x_bin
        )

    def act(self, state_dict):
        discrete_state = self.discretize_state(state_dict)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.zeros(len(self.actions))
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
