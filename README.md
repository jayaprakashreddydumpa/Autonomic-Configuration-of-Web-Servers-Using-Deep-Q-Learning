# CMPE295
Masters-Project
  
In this project, the main idea is to implement an algorithm based on deep reinforcement learning to automate the web server configuration. Reinforcement Learning is a promising mode which makes decisions based on the output of previous states. The combination of this model with a neural network further enhances its learning power which is a much more efficient approach for handling dynamic decision-making problems. So We plan on implementing a Deep-Q Network model on an apache server hosted on a virtual machine environment having different workloads to find the optimal parameters of the web server.


Project Requirements:
TensorFlow version 2.1
Keras
Python3.7 or latest
TensorBoard
Tqdm
Subprocess.run
Regex

Before running the code do the following steps:
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.7 get-pip.py
  
Once the latest python version and pip are installed. Install the project requirements
pip install tensorflow==2.1
pip install tensorboard
pip install subprocess.run
pip install keras
pip install regex
  
Place the webserver_config_dqn.py and dqn_test.py in the same directory
Run the command python3.7 webserver_config_dqn.py -- to train the model
Once the training is done, the models would be saved in "models/" subdirectory within the same directory
Run the command python3.7 dqn_test.py to test the final model
