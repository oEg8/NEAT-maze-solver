import pickle


class Toolbox:
    def save_model(network, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(network, f)
            f.close()


    def load_model(filename: str):
        return pickle.load(open(filename, 'rb')) 