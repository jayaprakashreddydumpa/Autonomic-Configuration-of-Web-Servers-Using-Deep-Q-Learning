import numpy as np
from keras.models import load_model
np.warnings.filterwarnings('ignore')
import subprocess
import webserver_config_dqn

def test(model, env):
    current_state = env.reset()
    action = np.argmax(model.predict(np.array(np.asarray(current_state)).reshape(-1, *np.asarray(current_state).shape)))
    print("The predicted action is: ",dummy.action_space[action])
    next_state, reward= env.step(action,current_state,0,True)
    current_state = next_state
    return reward



if __name__ == "__main__":
    env = webserver_config_dqn.Apache_environment()

    #Loading the model based on the best_reward obtained from the saved model
    LOAD_MODEL = "models/DQNAgentfinal.model"
    test_model = load_model(LOAD_MODEL)
    reward = test(test_model,env)
    print("The reward obtained is: ",reward)
