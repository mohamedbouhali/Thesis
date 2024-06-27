"""
Script responsible for generating new environments that are gradually more complex. Inspired from Evolutionry Algorithms
Implemented by Mohamed Bouhali.
Date: 28.06.2024
"""
import itertools
import os
import random
from itertools import combinations
import numpy as np
from beautifultable import BeautifulTable
from matplotlib import pyplot as plt
from crossover import harder_pos, bigger_size, more_obs
from complexity import get_complexity, convert_str2arr
from algorithm import DDPG
from env_from_xml import save_individual_in_temp
from feasibilty import  goal_in_obstacle, obstacle_not_in_table, coll
from mutation import mutate_individual
from param_to_xml import param_to_xml
from agent_trainer import Agent_Trainer
from algorithm.replay_buffer import goal_based_process, ReplayBuffer_Episodic
from envs import make_env
from xml_to_param import xml_to_params_and_structure
from test import Tester

class CurriculumLearner:
    def __init__(self, args):
        self.args=args
        self.population=[]  # list containing active individuals
        self.MAX_POP_SIZE=10 # the number of allowed obstacle in env is 10
        return

    def generate_valid_individual(self,env):
        """ generates the initial individuals in the population. These env are valid enviroments to make sure that the initial population is valid
        Return: List of individuals.
        """
        #extract the infos about the specific enviroment.
        d = os.path.dirname(os.path.abspath(__file__)) #we need the absolute path of directory containing curriculum.py file
        if self.args.env[:7]=="FetchRe":
            path_to_xml1 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/reach_a.xml')
            path_to_xml2 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/reach_b.xml')
            path_to_xml3 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/reach_c.xml')
            xml_string, object_bodies1 = self.xml_to_params(path_to_xml1, env)
            xml_string, object_bodies2 = self.xml_to_params(path_to_xml2, env)
            xml_string, object_bodies3 = self.xml_to_params(path_to_xml3, env)
        if self.args.env[:7]=="FetchPu":
            path_to_xml1 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/push_a.xml')
            path_to_xml2 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/push_b.xml')
            path_to_xml3 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/push_c.xml')
            xml_string, object_bodies1 = self.xml_to_params(path_to_xml1, env)
            xml_string, object_bodies2 = self.xml_to_params(path_to_xml2, env)
            xml_string, object_bodies3 = self.xml_to_params(path_to_xml3, env)
        if self.args.env[:7] == "FetchSl":
            path_to_xml1 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/slide_a.xml')
            path_to_xml2 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/slide_b.xml')
            path_to_xml3 = os.path.join(d, '../gym/gym/envs/robotics/assets/fetch/Initial_env/slide_c.xml')
            xml_string, object_bodies1 = self.xml_to_params(path_to_xml1, env)
            xml_string, object_bodies2 = self.xml_to_params(path_to_xml2, env)
            xml_string, object_bodies3 = self.xml_to_params(path_to_xml3, env)
        if self.args.env[:7] == "FetchPi":
            path_to_xml1 = os.path.join(d,'../gym/gym/envs/robotics/assets/fetch/Initial_env/pick_and_place_a.xml')
            path_to_xml2 = os.path.join(d,'../gym/gym/envs/robotics/assets/fetch/Initial_env/pick_and_place_b.xml')
            path_to_xml3 = os.path.join(d,'../gym/gym/envs/robotics/assets/fetch/Initial_env/pick_and_place_c.xml')
            xml_string, object_bodies1 = self.xml_to_params(path_to_xml1, env)
            xml_string, object_bodies2 = self.xml_to_params(path_to_xml2, env)
            xml_string, object_bodies3 = self.xml_to_params(path_to_xml3, env)

        #we Save the stuctre of the xml file, to use it afterward when saving the new generated xml files
        self.xml_structure=xml_string
        population=[]
        population.append(object_bodies1)
        population.append(object_bodies2)
        population.append(object_bodies3)
        self.population=population
        return population

    def crossover(self, args, env, father, mother):
        """
        Creates a new child from the given two parents, make the object is not present in the father or mother
        Parameters
        ----------
        father: dictionary containg all obstacles informations of the father
        mother:dictionary containg all obstacles informations of the mother

        Returns:dictionary containg all obstacles informations of the child
        -------
        """

        strategies = ['bigger_size', 'more_obs', 'harder_pos']
        strategy = random.choices(strategies, weights=[20, 60, 20], k=1)[0]#we select a strategy basd in the probilities we want. It makes some strategy more probable to be chosen
        print("The chosen strategy is: ", strategy)

        if strategy== 'bigger_size':
            child_env= bigger_size(self.args, env, father, mother)
        elif strategy== 'more_obs':
            child_env=  more_obs(self.args, env, father, mother)
        else: child_env= harder_pos(self.args, env, father, mother)

        return  child_env

    def mutate(self,args, individual):
        """
        Given an environment, the mutate function slightly modifies either the position or the size of the obstacle randomly.
        Return: Mutated obstacle.
        """
        mutation_rate=args.mutation_rate #mutation rate as given in the common.py file
        return mutate_individual(args, individual,mutation_rate)


    def calculate_complexity(self, args, env, object_infos):
        """
        Calculates the complexity of the enviroment
        Parameters
        ----------
        env : the enviroment (created by make_env() )
        xml_path : the path to the XML file

        Returns : The complexity
        -------
        """
        complexity = get_complexity(args, env, object_infos)
        print("complexity is: ", complexity)
        return complexity

    def acc_of_trained_agent_on_env(self,args,trained_agent):
        """
        Gives the accuracy of the given agent on a specific enviroment
        Parameters
        ----------
        args: THe args contains the specific enviroment to run the agent on. It's the same used in curriculum because for example if curriculum will run on fetch push, the agent will also run on fetch push
        trained_agent:the agent that will be used
        args._new_mujoco_path: contains the which index
        env_List=list of environments to try. Must be generated from make(args). Always change args.new_mujoco_path befre making new env. Save the xml in temp file, change new_mujoco_path, then make env_List
        -------
        Returns: accuracy
        Source: Test.py from HGG
        """
        env_List=[]
        for _ in range(50):
            env_List.append(make_env(args))

        test_rollouts=50
        acc_sum, obs = 0.0, []
        for i in range(test_rollouts):
            obs.append(goal_based_process(env_List[i].reset()))
        for timestep in range(args.timesteps):
            actions = trained_agent.step_batch(obs)
            obs, infos = [], []
            for i in range(test_rollouts):
                ob, _, _, info = env_List[i].step(actions[i])
                obs.append(goal_based_process(ob))
                infos.append(info)
        for i in range(test_rollouts):
            acc_sum += infos[i]['Success']

        steps = self.args.buffer.counter
        acc = acc_sum / test_rollouts
        print("The accuracy of the trained agent on the generated enviroment is", acc)
        return acc

    def is_feasible(self,args, env, individual):
        """
        Checks whether a generated enviroment is feasible and valid
        individual: dictionaries conating all existent bodies in the child
        Return: True if the child is valid anf False otherwise
        """
        # detetct collision between obsatcles and collision between object and obstacles
        key_pairs = combinations(individual.keys(), 2)
        if len(individual) >=2:
            for key1, key2 in key_pairs:
                element1 = individual[key1]
                element2 = individual[key2]
                center1 = convert_str2arr(element1.pos)
                size1 = convert_str2arr(element1.geom['geom']['size'])
                center2 = convert_str2arr(element2.pos)
                size2 = convert_str2arr(element2.geom['geom']['size'])

                result = coll(center1, size1, center2, size2)

                if result:
                    print("Enviroment is NOT valid")
                    return False




        # checks if the goal is inside an obstacle
        goal = env.goal
        goal = goal[:3]
        for _, obstacle_info in individual.items():
            object_center= convert_str2arr(obstacle_info.pos)
            object_size=convert_str2arr(obstacle_info.geom['geom']['size'])

            result= goal_in_obstacle(goal, object_center, object_size)
            if  result:
                print("Enviroment is NOT valid")
                return False


        # check if obstacles are in the table for push and slide.
        if args.env[:7]=="FetchRe" or  args.env[:7]=="FetchPi" :
            tab_center = np.array([1.3 ,0.75, 0.2])
            tab_size   = np.array([0.25 ,0.35, 0.2])
        elif args.env[:7]=="FetchPu":
            tab_center = np.array([1.3 ,0.75 ,0.2])
            tab_size = np.array([0.25 ,0.35 ,0.2])
        else:
            #obstacle_in_middle_for_slide()
            tab_center = np.array([1.32441906 ,0.75018422 ,0.2])
            tab_size = np.array([0.625 ,0.45 ,0.2])
        for _, obstacle_info in individual.items():
            object_center= convert_str2arr(obstacle_info.pos)
            object_size=convert_str2arr(obstacle_info.geom['geom']['size'])

            result= obstacle_not_in_table(tab_center, tab_size, object_center, object_size)
            if  result:
                print("Enviroment is NOT valid")
                return False


        print("Enviroment is valid")
        return True

    def save_individual_in_temp(self, args, xml_structure, inidvidual_object_infos,new_mujoco_path="temp"):
        env = save_individual_in_temp(args, xml_structure, inidvidual_object_infos,new_mujoco_path)
        return env

    def fitness(self,args,env, trained_agent,individual_as_env,individual_as_object_infos,k):
        """
        Calcultes the fitness of the given enviroment
        args: in args.new_mujoco_path contains the index of the xml file we want to test
        trained_agent: the trained agent which is used to get the accuracy of the agent on the enviroment
        individual_as_env: the enviroment as gym env
        individual_as_object_infos: the env as dictionary containg all obstacles and obejct in the env
        k: the weight used for the accuracy for fitness score
        """
        acc_fitness=self.acc_of_trained_agent_on_env(args,trained_agent)
        complexity_fitness=self.calculate_complexity(args,env,individual_as_object_infos)
        fitness= -k*acc_fitness +(1-k)*complexity_fitness
        #print("The fitness of the env is", fitness)
        return fitness

    def select(self,args,env,trained_agent,k, population_as_object_infos,current_generation):
        """
        Selects the individual with biggest fitness value and stores the corresponding xml file in the Env_generated_by_algo folder with the generation index
        """

        if not population_as_object_infos:
            print("All generated env in this generation are not valid, the training will be continued with th current population.")
            population_as_object_infos=self.population
            #return []
        def extract_fitness(element):
            return element[1]
        print("We are now selecting the best environments")
        ordered_individuals=[]
        best_fitness=-1
        for individual_as_object_infos in population_as_object_infos:
            individual_as_env=self.save_individual_in_temp(args,self.xml_structure, individual_as_object_infos)
            fitness = self.fitness(args,env,trained_agent,individual_as_env,individual_as_object_infos,k)
            ordered_individuals.append([individual_as_object_infos,fitness])
            if fitness>best_fitness:
                best_fitness=fitness
                best_individual_as_object_infos=individual_as_object_infos

        ordered_individuals= [element[0] for element in sorted(ordered_individuals, key=extract_fitness)]
        self.save_individual_in_temp(args, self.xml_structure, best_individual_as_object_infos,current_generation)
        return ordered_individuals

    def params_to_xml(self, xml_structure, object_info,folder_path,file_name):
        """Generate an XML file from the xml meta data and also the given parameters. It stores the xml file in the given folder.
        Input: xml_structure -> the structure of the xml file
               object_info -> the informations about the object that should be inserted in the xml file
               foler_path -> where to save the file
               Saves the file in the Env_generated_by_algo folder and increment push_0, push_1 ...
        """
        param_to_xml(xml_structure,object_info,folder_path,file_name )

    def xml_to_params(self, path_to_xml,env):
        """Parse XML file to extract environment parameters and return a dictionary for each component present in the mujuco enviroment
        , each entry conatins the name, size and position. It also returns the meta data for the xml file to be reconstructed without any changes
        Input:path to the xml file to parse
        Output: structure of the xml file and Dictionary of component in the mujoco xml file
        """
        xml_structure, object_info =xml_to_params_and_structure(path_to_xml,env)
        return xml_structure, object_info


    def run_evolution(self, args, env, env_test, trained_agent,buffer,current_generation):
       self.current_generation=args.new_mujoco_path
       #self.current_generation=current_generation
       self.table = BeautifulTable()

       """ 
       xml_structure, individual_as_object_infos =self.xml_to_params("/home/mohamed-bouhali/Thesis/gym/gym/envs/robotics/assets/fetch/Env_generated_by_algo/reach_0.xml",env)
       xml_structure_x, individual_as_object_infos_x = self.xml_to_params("/home/mohamed-bouhali/Thesis/gym/gym/envs/robotics/assets/fetch/Env_generated_by_algo/reach_x.xml", env)
       self.xml_structure=xml_structure

       individual_as_env=self.save_individual_in_temp(args, xml_structure, individual_as_object_infos)
       self.fitness(args, trained_agent,individual_as_env,individual_as_object_infos,1)
       print("In impossible env now")

       individual_as_env_x = self.save_individual_in_temp(args, xml_structure, individual_as_object_infos_x)
       self.fitness(args, trained_agent, individual_as_env_x, individual_as_object_infos_x, 1)


       population_as_object_infos=[]
       population_as_object_infos.append(individual_as_object_infos)
       population_as_object_infos.append(individual_as_object_infos_x) #try with reach_x to see if not selected and which file s saved in the next reach_1 file
       self.select( args, trained_agent, 1, population_as_object_infos, len(population_as_object_infos),current_generation)
       """
       print("Agent reached more than 0.7 Success Rate. Evolution started to generate new environments")
       if not self.population :
           self.generate_valid_individual(env)
           self.select(args,env,trained_agent,1, self.population,current_generation)
       else:
           parents = list(itertools.combinations(self.population, 2))
           offsprings=[]
           for p1,p2 in parents:
               child=self.crossover(args,env,p1,p2)
               child=self.mutate(args,child)
               if self.is_feasible(args,env,child):
                   offsprings.append(child)

           offsprings = self.select(args, env, trained_agent, 0.9, offsprings, current_generation)  # order the offsprings according to their fitness
           self.population+=offsprings
           self.population=self.population[-self.MAX_POP_SIZE :]


       args.new_mujoco_path=self.current_generation
       return




