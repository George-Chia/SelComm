import heapq
import os
import logging
import random
import sys
import time

import numpy as np
import rospy
import torch
import socket
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from torch.autograd import Variable
from collections import deque

from model.net import CNNPolicy,SelectorNet
from stage_world2 import StageWorld
from model.ppo import ppo_update_stage2, generate_train_data
from model.ppo import generate_action, transform_buffer
from model.dqn import get_influence_list,dqn_update
from model.utils2_6 import get_group_terminal, get_filter_index


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128 # 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 32
EPOCH = 4
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 12
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5
BANDWIDTH = 11

DQN_LR = 0.01
MEMORY_SIZE = 2000
MEMORY_THRESHOLD = 32
V_NETWORK_ITERATION = 100
STAY_SELECTOR = 5 #5


def run(comm, env, policy, policy_path, action_bound, optimizer,
        selector, target_selector, selector_optimizer,mse_selector, mode):
    rate = rospy.Rate(40)
    buff = []
    dqn_buff = deque()
    global_update = 0
    dvn_update_count = 0
    global_step = 0
    if env.index == 0:
        env.reset_world()


    for id in range(MAX_EPISODES):
        env.reset_pose()
        time.sleep(0.2)
        env.generate_goal_point()
        group_terminal = False
        ep_reward = 0
        dqn_reward = 0
        liveflag = True
        step = 0

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        position = np.asarray(env.get_position())
        state = [obs_stack, goal, speed, position]


        while not group_terminal and not rospy.is_shutdown(): # group_terminal guarantees the synchronization reset of all robotstage2-george-2comms

            state_list = comm.gather(state, root=0)
            position_list = comm.gather(position, root=0)
            if env.index == 0:
                if mode is 'position':
                    adj_list = get_adjacency_list(position_list,bandwidth=BANDWIDTH)
                    # print(adj_list)
                    # adj_list *= 0
                elif mode is 'random':
                    adj_list = [[0, 1, 1, 0, 0, 0,   1, 1, 1, 0, 0, 0],
                                [1, 0, 1, 0, 0, 0,   1, 1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0,   1, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1,   0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 0, 1,   0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 0,   0, 0, 0, 1, 1, 1],
                                [1, 1, 1, 0, 0, 0,   0, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0,   1, 0, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0,   1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1,   0, 0, 0, 0, 1, 1],
                                [0, 0, 0, 1, 1, 1,   1, 0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 1, 1,   0, 0, 0, 1, 1, 0],
                                ]
                    adj_list = np.array(adj_list)
                    # adj_list *= 0

                elif mode is 'dqn':
                    if step % STAY_SELECTOR ==0:
                        adj_list,_ = get_influence_list(state_list, selector = selector, bandwidth = BANDWIDTH)
                    # adj_list_position = get_adjacency_list(position_list, bandwidth=BANDWIDTH)
                    # print(adj_list)
                    # print(adj_list_position)
                    # # adj_list = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,]]
                    # # adj_list = np.array(adj_list)
                    # print('similarity ratio: ', ((adj_list * adj_list_position).sum())/(NUM_ENV*BANDWIDTH))
                    # print('--------------------')
                    # adj_list *= 0
                else:
                    traceback.print_exc()
                for state_, adj_ in zip(state_list,adj_list):
                    state_.append(np.asarray(adj_))
            # generate actions at rank==0
            v, a, logprob, scaled_action, all_attend_probs=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)
            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            if liveflag == True:
                env.control_vel(real_action)
                # rate.sleep()
                rospy.sleep(0.001)
                # get informtion
                r, terminal, result = env.get_reward_and_terminate(step)
                step += 1


            if liveflag == True:
                ep_reward += r
            if terminal == True:
                liveflag = False
            global_step += 1
            dqn_reward += r



            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            position_next = np.asarray(env.get_position())
            state_next = [obs_stack, goal_next, speed_next, position_next]


            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                position_next_list = comm.gather(position_next, root=0)
                if env.index == 0:
                    if mode is 'position':
                        adj_next_list = get_adjacency_list(position_next_list,bandwidth=BANDWIDTH)
                    elif mode is 'random':
                        adj_next_list = get_random_list(position_next_list, bandwidth=BANDWIDTH)
                    elif mode is 'dqn':
                        adj_next_list,_ = get_influence_list(state_next_list, selector=selector, bandwidth=BANDWIDTH)
                    else:
                        traceback.print_exc()
                    for state_next_, adj_next_ in zip(state_next_list, adj_next_list):
                        state_next_.append(np.asarray(adj_next_))
                last_v, _, _, _,_ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            #next_q for DQN
            state_next_list = comm.gather(state_next, root=0)
            if env.index == 0:
                if mode is 'dqn':
                    _, next_q = get_influence_list(state_next_list, selector=target_selector, bandwidth=BANDWIDTH)
                else:
                    next_q = None

            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            if step % STAY_SELECTOR == 0:
                dqn_reward = dqn_reward / STAY_SELECTOR
            dqn_reward_list = comm.gather(dqn_reward, root=0)
            # if r_list is not None:
            #     #print('before',r_list)
            #     r_list = r_list + 1.0 * np.matmul(r_list,all_attend_probs.T)
            terminal_list = comm.gather(terminal, root=0)
            terminal_list = comm.bcast(terminal_list, root=0)
            group_terminal = get_group_terminal(terminal_list, env.index)
            if env.index == 0:
                is_update = False
                buff.append((state_list, a, r_list, terminal_list, logprob, v, next_q))
                if step % STAY_SELECTOR == 1:
                    dqn_state_list = state_list
                    dqn_a = a
                    dqn_terminal_list = terminal_list
                    dqn_logprob = logprob
                    dqn_v = v


                if step % STAY_SELECTOR == 0:
                    dqn_next_q = next_q
                    dqn_buff.append((dqn_state_list, dqn_a, dqn_reward_list, dqn_terminal_list, dqn_logprob, dqn_v, dqn_next_q))
                    if len(dqn_buff) > MEMORY_SIZE:
                        dqn_buff.popleft()

                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, position_batch,adj_batch, a_batch, r_batch, d_batch, l_batch, v_batch, next_q_batch = \
                        transform_buffer(buff=buff)
                    filter_index = get_filter_index(d_batch)
                    # print d_batch.shape  [HORIZON, NUM_ENV]
                    # print len(filter_index)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, position_batch,adj_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch, next_q_batch,d_batch)
                    ppo_update_stage2(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory, filter_index=filter_index,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE, global_update = global_update)
                    is_update = True

                    buff = []
                    global_update += 1
                if len(dqn_buff) > HORIZON and mode is 'dqn' and (step % STAY_SELECTOR == 0):# and is_update==True:
                    s_batch, goal_batch, speed_batch, position_batch, adj_batch, a_batch, r_batch, d_batch, l_batch, v_batch, next_q_batch = \
                        transform_buffer(buff=dqn_buff)
                    filter_index = get_filter_index(d_batch)
                    dqn_memory = (s_batch, goal_batch, speed_batch, position_batch, r_batch, next_q_batch, d_batch)
                    dqn_update(selector=selector,selector_optimizer=selector_optimizer, mse_selector = mse_selector,
                            batch_size=BATCH_SIZE, memory=dqn_memory,filter_index=filter_index)
                    dvn_update_count += 1
                    if dvn_update_count % V_NETWORK_ITERATION == 0:
                        target_selector.load_state_dict(selector.state_dict())
                        print('update target selector')

            state = state_next
            position = position_next

            if step % STAY_SELECTOR == 0:
                dqn_reward = 0


        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/stage2_{}.pth'.format(global_update))
                if mode is 'dqn':
                    torch.save(selector.state_dict(), policy_path + '/dqn_stage2_{}.pth'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, %s,' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id, step-1, ep_reward, result))
        logger_cal.info(ep_reward)




def get_small_index(m, num):
    max_number = heapq.nsmallest(num, m)
    max_index = []
    for t in max_number:
        index = m.index(t)
        max_index.append(index)
        m[index] = 0
    return max_index


def get_adjacency_list(robots_position,bandwidth=3):
    num_robot = len(robots_position)
    communication_lists = []
    for robot_index in range(num_robot):
        dist_list = []
        for i in range(num_robot):
            dist = (robots_position[robot_index][0]-robots_position[i][0])**2 + (robots_position[robot_index][1]-robots_position[i][1])**2
            dist_list.append(dist)
        communication_index = get_small_index(dist_list,bandwidth+1)
        communication_lists.append(communication_index[1:])
    adj_list = np.zeros((num_robot,num_robot))
    for i in range(num_robot):
        for index in communication_lists[i]:
            adj_list[i][index] = 1
    return adj_list  # size:[12*3]

def get_random_list(robots_position,bandwidth=3):
    num_robot = len(robots_position)
    communication_lists = []
    for robot_index in range(num_robot):
        dist_list = []
        for i in range(num_robot):
            dist = random.random()
            dist_list.append(dist)
        communication_index = get_small_index(dist_list,bandwidth+1)
        communication_lists.append(communication_index[1:])
    adj_list = np.zeros((num_robot,num_robot))
    for i in range(num_robot):
        for index in communication_lists[i]:
            adj_list[i][index] = 1
    return adj_list  # size:[12*3]




if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]
    # torch.manual_seed(1)
    # np.random.seed(1)

    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        selector = SelectorNet(frames=LASER_HIST)
        target_selector = SelectorNet(frames=LASER_HIST)
        selector.cuda()
        target_selector.cuda()
        opt_selector = Adam(selector.parameters(), lr=DQN_LR)
        mse_selector = nn.MSELoss()


        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage2_3580.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')

        file_dqn = policy_path + '/start.pth'
        if os.path.exists(file_dqn):
            logger.info('####################################')
            logger.info('############Loading DQN Model###########')
            logger.info('####################################')
            dqn_state_dict = torch.load(file_dqn)
            selector.load_state_dict(dqn_state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start DQN Training###########')
            logger.info('#####################################')

        # selector_dict = selector.state_dict()
        # pretrained_dict = {k: v for k, v in state_dict.items() if k in selector_dict}
        # selector_dict.update(pretrained_dict)
        # selector.load_state_dict(selector_dict)
        # target_selector.load_state_dict(selector_dict)
    else:
        policy = None
        policy_path = None
        opt = None
        selector = None
        target_selector = None
        opt_selector = None
        mse_selector = None



    try:
        mode = 'position'
        # mode = 'dqn'
        # mode = 'random'
        print('#################MODE: ',mode,'#######################')
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt,
            selector=selector,target_selector=target_selector, selector_optimizer=opt_selector, mse_selector = mse_selector,mode = mode)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
