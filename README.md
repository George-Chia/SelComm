# SelComm

## Requirement

- Python 3.6
- [ROS, tested on Kinetic and Melodic](http://wiki.ros.org/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch 1.4](http://pytorch.org/)

## Setup

- #### Simulation environment

Please use the `stage_ros-add_pose_and_crash` package instead of the default package provided by ROS.

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

To train the random scenario, running the following command::

```
rosrun stage_ros_add_pose_and_crash stageros -g worlds/random.world
mpiexec -np NUM_ROBOTS(8, 16 or 24) python ppo_stage1.py
```

To train Stage2, modify the hyper-parameters in `ppo_stage2.py` as you like, and running the following command:

```
rosrun stage_ros_add_pose_and_crash stageros -g worlds/group-swap.world
mpiexec -np 12 python ppo_stage2.py
```

## References

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

@inproceedings{long2018towards,
  title={Towards optimally decentralized multi-robot collision avoidance via deep reinforcement learning},
  author={Long, Pinxin and Fan, Tingxiang and Liao, Xinyi and Liu, Wenxi and Zhang, Hao and Pan, Jia},
  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6252--6259},
  year={2018},
  organization={IEEE}
}
```
