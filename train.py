import numpy as np
import time
from common import get_args,experiment_setup ,experiment_setup_testing
import matplotlib.pyplot as plt
from envs import make_env
import csv

from learner import CurriculumLearner
from param_to_xml import param_to_xml
from xml_to_param import xml_to_params_and_structure
'''
Script for training using her or hgg. I used code from https://github.com/Stilwell-Git/Hindsight-Goal-Generation.git and also idea from https://github.com/erdiphd/COHER/tree/master to shift the environment when a specific success rate is reached.
'''
if __name__=='__main__':
    args = get_args()
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    if not args.use_curriculum:
        current_generation=args.number_of_generations -1
    else:
        current_generation=0
    generation = args.number_of_generations
    curriculum_learner = CurriculumLearner(args)

    #env2= experiment_setup_testing(args) #for simulating purposes
    # use interval, and energy based
    all_goals = []

    args.logger.summary_init(agent.graph, agent.sess)
    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('Episodes@green')
    args.logger.add_item('Timesteps')
    args.logger.add_item('TimeCost(sec)')

    # Algorithm info
    for key in agent.train_info.keys():
        args.logger.add_item(key, 'scalar')

    # Test info
    for key in tester.info:
        args.logger.add_item(key, 'scalar')

    args.logger.summary_setup()

    while current_generation<generation :
        print("Now in generation",current_generation)
        env = make_env(args)
        for epoch in range(args.epochs):
            for cycle in range(args.cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()
                explored_goals = learner.learn(args, env, env_test, agent, buffer)
                all_goals.extend(explored_goals)
                tester.cycle_summary()
                args.logger.add_record('Epoch', str(epoch) + '/' + str(args.epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(args.cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)
                args.logger.tabular_show(args.tag)
                args.logger.summary_show(buffer.counter)
                #same idea in https://github.com/erdiphd/COHER.git
                if args.logger.values["Success/interval"]  > 0.7:
                    break
            if args.logger.values["Success/interval"] > 0.7:
                break
            tester.epoch_summary()
        if args.logger.values["Success/interval"] > 0.7 :
            if current_generation!=(generation-1):
                curriculum_learner.run_evolution(args, env, env_test, agent, buffer, str(current_generation + 1))
            args.new_mujoco_path = str(int(args.new_mujoco_path)+1)
            current_generation+=1

    tester.final_summary()
        #if not args.use_curriculum:
            #break


