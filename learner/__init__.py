from .normal import NormalLearner
from .hgg import HGGLearner
from .curriculum import CurriculumLearner


learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
        'curriculum': CurriculumLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)
