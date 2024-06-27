import math
import random
from copy import deepcopy
import numpy as np
from complexity import convert_str2arr
from xml_to_param import xml_to_params_and_structure


def bigger_size(args, env, parent1,parent2):
    father = deepcopy(parent1) #we make a copy so that oroginal parent does not change
    mother = deepcopy(parent2) #same
    temp = father.pop('object0', None)# we do not want to modify the object, only the obstacles. So we remove the object so that it can not be randomly selected from the set of bodies in the environment, if the env is reach, None will be returned
    mother.pop('object0', None)
    # Get all the obstacles from parents
    obstacles_from_fath_moth = list(father.items()) + list(mother.items())
    # randomly choose obstacles for the child, divide the sum of obstacles in parent , then divide by 2 and round
    x = round(len(obstacles_from_fath_moth) / 2) #we take the averag number of obstacles in the parents
    obstacles_in_child = random.sample(obstacles_from_fath_moth, x)
    obstacles_in_child = [(f'obstacle{index}', value) for index, (key, value) in enumerate(obstacles_in_child)]#avoid merging items with same keys
    # we add the index i , so that obstacles with same number in parents are distinguished. For example obstacle0 in parent1 and obstacle1 in parent2 can be distingished unless we change the number to obstacle0 and obstacle1
    obstacles_in_child = dict(obstacles_in_child)

    # randomly choose one obtacle in child to make it bugger
    obs_to_change = random.choice(list(obstacles_in_child.values()))
    # extract the size of the obstacle
    size_of_obs_to_change = obs_to_change.geom['geom']['size']
    size_of_obs_to_change = convert_str2arr(size_of_obs_to_change)
    # randomly choose an achse in which the size will bet bigger
    achse_list = [0, 1, 2]
    if args.env[:7] == "FetchPu" or args.env[:7] == "FetchSl":
        achse_list = [0, 1]
    achse = random.choice(achse_list)# randomly choose an axis,acording to which the size be changed
    # multipy by small factor choosen from the gaussian distribution
    size_of_obs_to_change[achse] *= (random.gauss(1.1, 0.05)) #multiply the size by the small value
    # save the new size in the obstacle info
    obs_to_change.geom['geom']['size'] = ' '.join(map(str, size_of_obs_to_change.tolist())) # we convert the arrray to string to save later in XML file
    if args.env[:7] != "FetchRe":
        obstacles_in_child['object0'] = temp #add the popped object
    return  obstacles_in_child


def more_obs(args, env, father1,mother1):
    father = deepcopy(father1)
    mother = deepcopy(mother1)
    temp = father.pop('object0', None)
    mother.pop('object0', None)
    # Get all the obstacles from mother and father
    #obstacles_from_fath_moth = list(father.items()) + list(mother.items())
    f=len(father)
    m=len(mother)
    sum = f+m
    if f == m:
        x = f + 1 #if both parent have same number of obstacle. we just increment by one
    else:
        x = sum / 2
        if not x.is_integer():# if one parent has n obstacles and the other has n+2, then we take n+1 as result
            x = math.ceil(x + 0.01) # we add 0.01 so that the resilt is bigger than average
    #Now x is the number of obstacles in the child
    num_obs_moth=math.floor(x/2) #we round down
    num_obs_fath=math.ceil(x/2) #round up
    #we round ,so that the sum of num_obs_moth+num_obs_fath is equal to the original number x
    obs_from_moth=random.sample(list(mother.items()), num_obs_moth) #get half of obstacles from  father
    obs_from_fath=random.sample(list(father.items()), num_obs_fath) #get half of obstacles from  mother
    obstacles_in_child=obs_from_fath+obs_from_moth
    obstacles_in_child = [(f'obstacle{i}', value) for i, (key, value) in
                          enumerate(obstacles_in_child)]  # avoid merging items with same keys
    # we add the index i , to make it possible to differentiate between obstacles from each other
    obstacles_in_child = dict(obstacles_in_child)
    if args.env[:7] != "FetchRe":
        obstacles_in_child['object0'] = temp
    return obstacles_in_child

def harder_pos(args,env,father1,mother1):
    father=deepcopy(father1) # we create a copy so that teh original parent is not modified.
    mother=deepcopy(mother1)

    temp = father.pop('object0', None)
    mother.pop('object0', None)
    # Get all the obstacles from mother and father
    obstacles_from_fath_moth = list(father.items()) + list(mother.items())
    # randomly choose obstacles for the child, divide the sum of obstacles in parent , then divide by 2 and round
    x = round(len(obstacles_from_fath_moth) / 2)
    obstacles_in_child = random.sample(obstacles_from_fath_moth, x)
    obstacles_in_child = [(f'obstacle{index}', value) for index, (key, value) in
                          enumerate(obstacles_in_child)]  # avoid merging items with same keys
    #same as before, we index each obstacle to make it possible to differintiate between obstacles with same keys
    obstacles_in_child = dict(obstacles_in_child)
    #crandomly choose the obstacle to chnage its position
    obs_to_change = random.choice(list(obstacles_in_child.values()))
    pos_of_obs_to_change = obs_to_change.pos #extract the position
    pos_of_obs_to_change = convert_str2arr(pos_of_obs_to_change)
    achse_list = [0, 1, 2]
    if args.env[:7] == "FetchPu" or args.env[:7] == "FetchSl":
        achse_list = [0, 1]
    achse = random.choice(achse_list)
    pos_of_obs_to_change=pos_of_obs_to_change

    goal = env.goal
    goal_pos = goal[:3]# extract the goal position from gym environment
    if args.env[:7] == "FetchRe":  # extract the pos from gym
        object_position = env.initial_gripper_xpos[:3]
    else:
        object_position = (env.sim.data.get_joint_qpos('object0:joint'))[:3]

    value_to_add = 0
    point_between_goal_obj=object_position+goal_pos
    point_between_goal_obj=point_between_goal_obj/2 # we take the center of the segment between the goal and the object
    distance_obstacle_mid_point=np.linalg.norm(pos_of_obs_to_change-point_between_goal_obj)
    x=distance_obstacle_mid_point
    temp_new_pos_obstacle = np.copy(pos_of_obs_to_change)

    #randomly change the positon of the obstacles for 100 times, then calculate the distacne between the pos of obstacle and the mpoint in the center of the line between the goal and the object to move
    for _ in range(100):
        value = random.uniform(-0.03, 0.03)
        temp_new_pos_obstacle[achse]+= value
        dis=np.linalg.norm(point_between_goal_obj-temp_new_pos_obstacle)
        if dis<distance_obstacle_mid_point: #if the new coordinates are closer to the center of the segment, we save the new value
            value_to_add=value
            distance_obstacle_mid_point=dis
        temp_new_pos_obstacle[achse]-= value

    pos_of_obs_to_change[achse]+=value_to_add #this is value that minimizes the distance between the obstacle and the center of the segment between goal and object
    obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist()))#convert back to string to save later in XML file
    if args.env[:7] != "FetchRe":
        obstacles_in_child['object0'] = temp #add the popped object
    return obstacles_in_child




