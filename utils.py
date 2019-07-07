import pickle

def load_data(file):
    try:
        with open(file, 'rb') as infile:
            obj = pickle.load(infile)

        return obj

    except FileNotFoundError:
        print("[ERROR] File not found. {}".format(file))
        return None

