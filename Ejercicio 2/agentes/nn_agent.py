from agentes.dq_agent import QAgent
import numpy as np
import tensorflow as tf

class NNAgent(QAgent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_birds_q_expert_model.h5'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def act(self, state_dict):
        discrete_state = self.discretize_state(state_dict)
        discrete_state = np.array(discrete_state).reshape(1,-1)

        q_value = self.model.predict(discrete_state, verbose=0)
        action = int(q_value > 0.5)
        if action == 1:
            return 119
        return None
