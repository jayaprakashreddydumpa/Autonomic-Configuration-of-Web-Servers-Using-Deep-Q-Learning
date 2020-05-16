<h1>Masters-Project</h1>

<p>In this project, the main idea is to implement an algorithm based on deep reinforcement learning to automate the web server configuration. Reinforcement Learning is a promising mode which makes decisions based on the output of previous states. The combination of this model with a neural network further enhances its learning power which is a much more efficient approach for handling dynamic decision-making problems. So We plan on implementing a Deep-Q Network model on an apache server hosted on a virtual machine environment having different workloads to find the optimal parameters of the web server.</p>

Project Requirements: <br>
•	TensorFlow version 2.1 <br>
•	Keras <br>
•	Python3.7 or latest <br>
•	TensorBoard Tqdm <br>
•	Subprocess.run <br>
•	Regex <br>
<br>
Before running the code do the following steps: <br>
•	sudo apt update <br>
•	sudo apt install software-properties-common <br>
•	sudo add-apt-repository ppa:deadsnakes/ppa <br>
•	sudo apt update <br>
•	sudo apt install python3.7 <br>
•	curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py <br>
•	python3.7 get-pip.py <br>
<br>
Once the latest python version and pip are installed. Install the project requirements <br>
•	pip install tensorflow==2.1 <br>
•	pip install tensorboard <br>
•	pip install subprocess.run <br>
•	pip install keras <br>
•	pip install regex <br>
<br>
Place the webserver_config_dqn.py and dqn_test.py in the same directory <br>
Run the command python3.7 webserver_config_dqn.py -- to train the model <br>
Once the training is done, the models would be saved in "models/" subdirectory within the same directory <br>
Run the command python3.7 dqn_test.py to test the final model <br>

