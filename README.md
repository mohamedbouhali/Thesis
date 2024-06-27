# Bachelor's Thesis: Curriculum learning for robot manipulation tasks through environment shifts

This is my implementation. It's based on HGG (https://github.com/Stilwell-Git/Hindsight-Goal-Generation) Many functions are used from here.


## Requirements 
First create a conda environment. Then, in this env install :
1. Python 3.6.9
2. MuJoCo == 1.50.1.68
3. TensorFlow >= 1.8.0
4. BeautifulTable == 0.7.0
5. gym < 0.22
6. lxml
7. trimesh
8. rtree
9. pip install python-fcl

## Running Commands

Run the following commands to reproduce our main results shown in section 5.1.

```bash
python train.py 

```

If you encounter a problem that says gym is not found, use "pip install -e gym"
