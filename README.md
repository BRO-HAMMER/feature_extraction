# feature_extraction

from BROFEATURES import KPFeatures

class = KPFeatures(detector="SURF", extractor="SURF", matcher="BruteForce", flann_params=None, dictsize=500, model_dir="", model_name=None, verbose=True)

class.load_model(directory, name, verbose=True)

class.process_img(imgfile, verbose=True)

class.process_batch(directory="", types=("png", "jpg"), name="batch", autosave=False, verbose=True)

class.save_model(directory, name)

class.best_match(imgfile, ratio=0.75, visualize=2, verbose=True)