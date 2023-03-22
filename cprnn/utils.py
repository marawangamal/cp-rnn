import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:  # Overwrites any existing file.
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        pickle.load(inp_file)
