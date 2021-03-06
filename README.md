# SelComm_Stage

This repository contains the codes for our paper, which is in submission to RA-L and IROS 2021.

This is the implementation based on Stage_ROS. For Webots version, please refer to another [repository](https://github.com/George-Chia/SelComm_Webots).

## Requirement

- Python 3.6
- [ROS Melodic](http://wiki.ros.org/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch 1.4](http://pytorch.org/)

## Setup

- #### Simulation environment

Install the `stage_ros-add_pose_and_crash` package.

```shell
mkdir -p catkin_ws/src
cp stage_ros-add_pose_and_crash catkin_ws/src
cd catkin_ws
catkin_make
source devel/setup.bash
```

- #### Use Python3 and tf in ROS

```shell
sudo apt-get install python3-dev 
mkdir -p catkin_ws_py3/src
cd catkin_ws_py3/src  
git clone https://github.com/ros/geometry 
git clone https://github.com/ros/geometry2 
cd .. 
virtualenv -p /usr/bin/python3 venv 
source venv/bin/activate 
pip install catkin_pkg pyyaml empy rospkg numpy 
catkin_make --cmake-args -DPYTHON_VERSION=3.6
source devel/setup.bash
```



## How to train

To train the model in the random scenario, running the following command::

```
rosrun stage_ros_add_pose_and_crash stageros -g worlds/random.world
mpiexec -np NUM_ROBOTS(8, 16 or 24) python ppo_stage1.py
```

To train the model in the group swap scenario, running the following command:

```
rosrun stage_ros_add_pose_and_crash stageros -g worlds/group-swap.world
mpiexec -np 12 python ppo_stage2.py
```

## References

 The authors thank Liu for the open sourced code.

```
@misc{Tianyu2018,
  author = {Tianyu Liu},
  title = {Robot Collision Avoidance via Deep Reinforcement Learning},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Acmece/rl-collision-avoidance.git}},
  commit = {7bc682403cb9a327377481be1f110debc16babbd}
}
```
