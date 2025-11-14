# AI for Robotics
Designing a lane-keeping algorithm for an autonomous car.

The algorithm uses a basic PID controller that applies PCA to compute its heading angle and incorporates Reinforcement Learning to improve the overall performance.

## Run Simulations

### Clone the repo

Clone the current version of the repo and clone the gymnasium

```
$ git clone git@github.com:Bongni/AI-for-Robotics.git
```
```
$ cd AI-for-Robotics
```
```
$ git clone git@gitlab.unimelb.edu.au:ai4r/ai4r-gym.git
```

###

The final layout should look like this

```
AI-for-Robotics/
├── ai4r-gym/
├── ai4r-system/
├── config/
├── models/
├── results/
├── PCA_RL.ipynb
├── PCA.ipynb
├── PID_RL.ipynb
├── policy_node_RL.py
├── policy_node.py
└── README.md
```


## Load policy onto car

Connect via ssh

`ssh asc@asc##-rpi.eng.unimelb.edu.au`

### Clone the repo

Clone the current version of the repo

```
git clone git@github.com:Bongni/AI-for-Robotics.git
```

### Pull the most recent version

Simply pull the most recent version of the main branch after checking that there are no unsaved changes

```
cd AI-for-Robotics
```
```
git status
```
```
git checkout main
```
```
git pull
```

Move the `policy_node.py` to following folder on the car

```
~/ai4r-system/ros2_ws/src/ai4r_pkg/scripts/
```

### Build

```
cd ai4r-system/ros2_ws/
```
```
colcon build --symlink-install
```

### Connect with Foxglove

Open another ssh connection

`ssh asc@asc##-rpi.eng.unimelb.edu.au`

and launch foxglove

`launch_foxglove`

Open connection and enter `ws://10.41.190.###:1234`, where you need to use the ethernet port = car number + 40.

Open two more terminals and connect them via ssh, then run each command

```
launch_policy
```

```
launch_nodes traxxas detector imu tof lidar
```

## Resources

Silver, T., Allen, K., Tenenbaum, J., and Kaelbling, L. (2019). Residual policy learning. [https://arxiv.org/pdf/1812.06298]
