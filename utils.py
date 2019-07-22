import pickle
import os


def load_data(file):
    try:
        with open(file, 'rb') as infile:
            obj = pickle.load(infile)

        return obj

    except FileNotFoundError:
        print("[ERROR] File not found. {}".format(file))
        return None


# create a folder if it doesn't exist, else pass
def create_folder(path):
    # make sure there is an output directory
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("COULDN'T CREATE DIRECTORY:", path)


# move a file to a destination folder
def move_file(currentpath, destination):
    name = os.path.basename(currentpath)
    # make sure the destination exists
    create_folder(destination)
    # move the file preserving the name
    os.rename(currentpath, os.path.join(destination, name))
