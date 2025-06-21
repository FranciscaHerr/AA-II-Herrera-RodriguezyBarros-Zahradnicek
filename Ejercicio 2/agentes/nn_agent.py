from agentes.dq_agent import QAgent
import numpy as np
import tensorflow as tf

class NNAgent(QAgent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def act(self, state_dict):
        discrete_state = self.discretize_state(state_dict)
        normalized_state = np.array(discrete_state) / 9
        normalized_state = normalized_state.reshape(1,-1) 

        q_values = self.model.predict(normalized_state)
        action = np.argmax(q_values)
        return action
