import pickle

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix():
    pass


def load(args):
    try:
        with open(args.cache_path) as f:
            util_matrix = pickle.load(f)
    except:
        print("Utilization matrix not found, creating one now")
        util_matrix = create_util_matrix()
        print('Utilization matrix created')


def get_utilization(route):
    if util_matrix is None:
        raise(ValueError('You have not yet loaded the utilization matrix'))
    else:
        # Eventually replace with code to return utilization
        pass
