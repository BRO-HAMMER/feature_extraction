# feature_extraction
**Import class**

from BROFEATURES import KPFeatures

**Construct class with detector, extractor and matcher 
(can be initialized from an existing model).**

class = KPFeatures(detector="SURF", extractor="SURF", 
matcher="BruteForce", flann_params=None, dictsize=500, 
model_dir="", model_name=None, verbose=True)

**Use diagnose to evaluate number of matches for a 
sample against a dataset. Play with different detectors, 
extractors and matchers, and tweak parameters like dictsize 
to find the best configuration for your dataset.** 

class.diagnose(self, sample, directory, types=("png", "jpg"), 
r_range=(0.5, 0.7, 0.9)):

class.load_model(directory, name, verbose=True)

**Generate a model from an image dataset.**
class.process_batch(directory, types=("png", "jpg"), name="batch", 
autosave=False, verbose=True)

class.save_model(directory, name)

**Once a model has been created, find the best match to a 
sample image.**
class.best_match(imgfile, ratio=0.75, visualize=2, verbose=True)