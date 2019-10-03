import pickle

util_matrix = None


def create_util_matrix():
    pass


def load():
    try:
        with open('..//models//time_models//time_matrix.pkl') as f:
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
