import pickle

time_matrix = None


def create_time_matrix():
    pass


def load(args):
    try:
        with open('..//models//time_models//time_matrix.pkl') as f:
            time_matrix = pickle.load(f)
    except:
        print("Time matrix not found, creating one now")
        time_matrix = create_time_matrix()
        print('Time matrix created')


def get_time(route):
    return 40 + 0.5 * sum(route)
