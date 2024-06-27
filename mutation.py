import random
from copy import deepcopy
from complexity import  convert_str2arr


def mutate_individual(args, child,mutation_rate):
    if random.random() <mutation_rate:# the probability of a mutation is a hyerparmeter mutation_rate fixed in the common.py file
        strategies = ['mutate_size', 'mutate_pos']
        strategy = random.choice(strategies) # we select randomly whether to change the position or the size of a obstacle


        #This part is similar to the crossover implementation
        child = deepcopy(child)
        temp = child.pop('object0', None)
        obs_to_change = random.choice(list(child.values())) #we randomly select an obsatcle to mutate
        #we exctract the size of this obs
        size_of_obs_to_change = obs_to_change.geom['geom']['size'] # get the string from the geom attribute of BodyInfo class
        size_of_obs_to_change = convert_str2arr(size_of_obs_to_change) #we convert it to a numpy array
        #secondly we exctract the position of this obs
        pos_of_obs_to_change = obs_to_change.pos
        pos_of_obs_to_change = convert_str2arr(pos_of_obs_to_change)

        if args.env[:7] == "FetchPu":
            if strategy == 'mutate_size':
                size_of_obs_to_change[0] *= (random.gauss(1.1, 0.05)) #we multiply the current size on x axis by small epsilon
                obs_to_change.geom['geom']['size'] = ' '.join(map(str, size_of_obs_to_change.tolist())) #convert the array back to a string needed for saving the value in xml file
            if strategy == 'mutate_pos':
                achse_list = [0, 1]
                achse = random.choice(achse_list)# we choose a random axis
                value = random.uniform(-0.03, 0.03) # we choose a random value from a normal distribution
                pos_of_obs_to_change[achse] += value # the pos is added the value in the corresponsing axis
                obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist())) #we convert back to string

        #we do the same for other environments
        if args.env[:7] == "FetchPi":
            if strategy == 'mutate_size':
                achse_list = [0, 2]
                achse = random.choice(achse_list)
                increase = random.gauss(1.1, 0.05) #random value from a gaussian distribution
                size_of_obs_to_change[achse] *= increase
                obs_to_change.geom['geom']['size'] = ' '.join(map(str, size_of_obs_to_change.tolist()))
                if achse == 2:
                    pos_of_obs_to_change[achse] += (size_of_obs_to_change[achse]) * (1 - (1 / increase))
                    obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist()))
            if strategy == 'mutate_pos':
                achse_list = [0, 1]
                achse = random.choice(achse_list)
                value = random.uniform(-0.03, 0.03) #random value from a normal distribution
                pos_of_obs_to_change[achse] += value
                obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist()))

        if args.env[:7] == "FetchRe":
            if strategy == 'mutate_size':
                achse_list = [0, 2]
                achse = random.choice(achse_list)
                increase = random.gauss(1.1, 0.05)
                size_of_obs_to_change[achse] *= increase
                obs_to_change.geom['geom']['size'] = ' '.join(map(str, size_of_obs_to_change.tolist()))
                if achse == 2:
                    pos_of_obs_to_change[achse] += (size_of_obs_to_change[achse]) * (1 - (1 / increase))
                    obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist()))
            if strategy == 'mutate_pos':
                achse_list = [0, 1]
                achse = random.choice(achse_list)
                value = random.uniform(-0.03, 0.03)
                pos_of_obs_to_change[achse] += value
                obs_to_change.pos = ' '.join(map(str, pos_of_obs_to_change.tolist()))

        #not implemented because curriculum learning is not suitable for FetchSlide
        if args.env[:7] == "FetchSl":
            pass
        #We add the object if the env is not FetchReach
        if args.env[:7] != "FetchRe":
            child['object0'] = temp

    return  child
