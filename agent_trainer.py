import copy
import time
from algorithm import create_agent
from algorithm.replay_buffer import ReplayBuffer_Episodic, Trajectory, goal_based_process
from envs import make_env, clip_return_range, Robotics_envs_id
from learner import HGGLearner, NormalLearner
from test import Tester
from utils.os_utils import get_logger
class Agent_Trainer:
    """
    This class is responsible for training the agent which will be used to calculate the fitnessof a generated enviroment
    The argument are the same arguments used in train.py but it's trained with HER
    Return: The trained agent and the Tester
    Source: The code is from HGG repository (train.py) with normal learner
    """
    def __init__(self, args):
        self.args=args

    def test_acc(self,args, key, env_List, agent, logger,buffer,tester):
        test_rollouts = 50
        acc_sum, obs = 0.0, []
        for i in range(test_rollouts):
            obs.append(goal_based_process(env_List[i].reset()))
        for timestep in range(self.args.timesteps):
            actions = agent.step_batch(obs)
            obs, infos = [], []
            for i in range(test_rollouts):
                ob, _, _, info = env_List[i].step(actions[i])
                obs.append(goal_based_process(ob))
                infos.append(info)
        for i in range(test_rollouts):
            acc_sum += infos[i]['Success']

        steps = buffer.counter
        acc = acc_sum / test_rollouts
        tester.acc_record[key].append((steps, acc))
        logger.add_record('Success/' + key, acc)

    def train_agent_her(self,args,agent, env, env_test, buffer,epochs, cycles):
        tester = Tester(args)

        print("Started to train the agent using HER")
        intermediate_goal_list=[]
        for epoch in range(epochs): #epochs
            for cycle in range(cycles):#cycles
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()
                #normal
                for _ in range(50):
                    obs = env.reset()
                    current = Trajectory(obs)
                    for timestep in range(args.timesteps):
                        action = agent.step(obs, explore=True)
                        obs, reward, done, _ = env.step(action)
                        if timestep == args.timesteps - 1: done = True
                        current.store_step(action, obs, reward, done)
                        if done: break
                    buffer.store_trajectory(current)
                    agent.normalizer_update(buffer.sample_batch())

                    if buffer.steps_counter >= args.warmup:
                        for _ in range(args.train_batches):
                            info = agent.train(buffer.sample_batch())
                            args.logger.add_dict(info)
                        agent.target_update()
                    intermediate_goal_list.append(obs["achieved_goal"])
                #normal




                env_List = []
                for _ in range(50):
                    env_List.append(make_env(args))
                self.test_acc(args, args.goal, tester.env_List, agent, args.logger, buffer ,tester)

                args.logger.add_record('Epoch', str(epoch) + '/' + str(epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)
                args.logger.tabular_show("")
                args.logger.summary_show(buffer.counter)
                if tester.acc_record[args.goal][-1][1] > 0.7:
                    break
            tester.epoch_summary()
            if tester.acc_record[args.goal][-1][1] > 0.7:
                break

        tester.final_summary()
        intermediate_goal_list.append(obs["desired_goal"])
        print("Finished training the agent using HER ")
        args.agent = agent
        args.buffer = buffer
        return agent,buffer,intermediate_goal_list


    def train_agent_her_test(self, args, agent, env, env_test, buffer, epochs, cycles):
        learner = NormalLearner(args)
        tester = Tester(args)

        print("Started to train the agent using HER test")
        intermediate_goal_list = []
        for epoch in range(epochs):  # epochs
            for cycle in range(cycles):  # cycles
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()

                learner.learn(args, env, env_test, agent, buffer)

                env_List = []
                for _ in range(50):
                    env_List.append(make_env(args))
                self.test_acc(args, args.goal, tester.env_List, agent, args.logger, buffer, tester)

                args.logger.add_record('Epoch', str(epoch) + '/' + str(epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)
                args.logger.tabular_show("")
                args.logger.summary_show(buffer.counter)
                if tester.acc_record[args.goal][-1][1] > 0.7:
                    break
            tester.epoch_summary()
            if tester.acc_record[args.goal][-1][1] > 0.7:
                break

        tester.final_summary()
        #intermediate_goal_list.append(obs["desired_goal"])
        print("Finished training the agent using HER test")
        args.agent = agent
        args.buffer = buffer
        return agent, buffer, intermediate_goal_list


    def train_agent_hgg(self,args,agent, env, env_test, buffer,epochs, cycles):
        learner=HGGLearner(args)
        intermediate_goal_list=[]
        tester = Tester(args)

        print("Started to train the agent using HGG")

        for epoch in range(epochs):
            for cycle in range(cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()
                learner.learn(args, env, env_test, agent, buffer)

                env_List = []
                for _ in range(50):
                    env_List.append(make_env(args))
                self.test_acc(args, args.goal, tester.env_List, agent, args.logger, buffer, tester)

                args.logger.add_record('Epoch', str(epoch) + '/' + str(epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)
                args.logger.tabular_show("")
                args.logger.summary_show(buffer.counter)
                if tester.acc_record[args.goal][-1][1] > 0.7:
                    break
            tester.epoch_summary()
            if tester.acc_record[args.goal][-1][1] > 0.7:
                break

        tester.final_summary()
        #intermediate_goal_list.append(obs["desired_goal"])     get obs from hgg file
        print("Finished training the agent using HGG ")
        args.agent = agent
        args.buffer = buffer
        return agent, buffer, intermediate_goal_list

        return agent,buffer ,intermediate_goal_list



