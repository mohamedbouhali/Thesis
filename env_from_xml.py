import os
import mujoco_py
from envs import make_env
from param_to_xml import param_to_xml


def save_individual_in_temp(args, xml_structure, inidvidual_object_infos,new_mujoco_path):
   """
   Creates a gym enviroment based on the new_mujoco_path index in args and saves the file in _temp.xml file in Env_generated_by_algo folder
   xml_structure: string containg the XML file without bodies
   inidvidual_object_infos: dictionary containg all bodies information that the new enviroment will have
   """
   args.new_mujoco_path = new_mujoco_path
   if args.env[:7]=="FetchRe":
      file_name="reach_"
   if args.env[:7]=="FetchPu":
      file_name="push_"
   if args.env[:7]=="FetchPi":
      file_name="pick_and_place_"
   if args.env[:7]=="FetchSl":
      file_name="slide_"

   file_name+= new_mujoco_path+".xml"
   directory = os.path.dirname(os.path.abspath(__file__))
   folder_path = os.path.join(directory, "../Thesis/gym/gym/envs/robotics/assets/fetch/Env_generated_by_algo")
   param_to_xml(xml_structure, inidvidual_object_infos,folder_path ,file_name)
   env = make_env(args)
   return env