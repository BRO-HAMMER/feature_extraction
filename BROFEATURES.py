import cv2
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create
from imutils import resize
from glob import glob
from os import path
import pickle
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import feature
from utils import load_data


class KPFeatures:

    def __init__(self, detector="SURF", extractor="SURF", matcher="BruteForce", flann_params=None,
                 dictsize=500, resized_h=600, model_dir="", model_name=None, verbose=True):
        """
        :param detector: "BRISK", "DENSE", "FAST", "GFTT", "HARRIS", "MSER", "ORB", "SIFT", "SURF", "STAR"
        :param extractor: "SIFT", "SURF", "BRIEF", "ORB", "BRISK", "FREAK"
        :param matcher: "FLANN", else "BruteForce" will be used
        """
        # get parameters
        if model_name is not None:
            config_file = path.join(model_dir, model_name+"_config.json")
            try:
                with open(config_file, "r") as infile:
                    self.config = json.load(infile)
            except FileNotFoundError:
                print("[ERROR] Model configuration file not found.")
                model_name = None

        if model_name is None:
            self.config = {"detector": detector,
                           "extractor": extractor,
                           "matcher": matcher,
                           "flann_params": flann_params,
                           "dictsize": dictsize,
                           "resized_h": resized_h}

        # set maximum size for image processing
        self.resizedH = self.config["resized_h"]

        # initialize detector and extractor
        self.detector = FeatureDetector_create(self.config["detector"])
        self.extractor = DescriptorExtractor_create(self.config["extractor"])

        # intialize matcher, KNN trainer and BOW extractor
        if self.config["matcher"] == "FLANN":
            if self.config["flann_params"] is None:
                self.config["flann_params"] = dict(algorithm=1, trees=5)

            self.matcher = cv2.FlannBasedMatcher(self.config["flann_params"], {})
        else:
            # use brute force matcher
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # initialize bow and svm
        self.BOWtrainer = cv2.BOWKMeansTrainer(self.config["dictsize"])
        self.BOWextractor = cv2.BOWImgDescriptorExtractor(self.extractor, self.matcher)

        # initialize model objects
        self.vocabulary = None
        self.svm = None
        self.dictionary = None

        # get objects from file if indicated
        if model_name is not None:
            self.load_model(model_dir, model_name, verbose)

    def load_model(self, directory, name, verbose=True):

        # load vocab, svm and dictionary from files
        vocab_file = path.join(directory, name + "_vocab.pickle")
        self.vocabulary = load_data(vocab_file)
        self.BOWextractor.setVocabulary(self.vocabulary)

        svm_file = path.join(directory, name + "_svm.xml")
        self.svm = cv2.ml.SVM_load(svm_file)

        dictionary_file = path.join(directory, name + "_dict.pickle")
        self.dictionary = load_data(dictionary_file)

        if self.vocabulary is not None and self.dictionary is not None and self.svm is not None:
            if verbose:
                print("[INFO] Model successfully loaded.")

        else:
            print("[ERROR] Model loading failed. Please load one using "
                  ".load_model(directory, name) or train a new one.")
            self.vocabulary, self.dictionary, self.svm = None, None, None

    def process_img(self, imgfile, verbose=True):

        # load the images, resize it and convert to grayscale
        image = cv2.imread(imgfile)

        if image.shape[0] > self.resizedH:
            image = resize(image, height=self.resizedH)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints
        kps = self.detector.detect(gray)

        (kps, features) = self.extractor.compute(gray, kps)

        if verbose:
            print(imgfile, "# of keypoints: {}".format(len(kps)))

        return kps, features

    def process_batch(self, directory, name="feats", types=("png", "jpg", "jpeg", "tif"),
                      autosave=False, verbose=True):
        # generate a model from a dataset of images contained in {directory}
        images = []
        for t in types:
            found = glob(path.join(directory,  "*." + t))
            images.extend(found)

        # construct vocabulary
        self.train_BOW(images, verbose=verbose)

        traindata, trainlabels,  = [], []
        dictionary = {}
        for n, im in enumerate(images):
            # relate labels to image paths and store that information in the dictionary
            dictionary[n] = im

            # upload the image, resize it and convert it to grayscale
            image = cv2.imread(im)

            if image.shape[0] > self.resizedH:
                image = resize(image, height=self.resizedH)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # create train data from BOW vectors
            data = self.BOWextractor.compute(gray, self.detector.detect(gray))
            if data is not None:
                traindata.extend(data)
                trainlabels.append(n)

        # train svm model and store dictionary data
        self.train_SVM(traindata, trainlabels)
        self.dictionary = dictionary

        if autosave:
            self.save_model(directory, name)

        if verbose:
            print("[INFO] Model created from database containing {} images.".format(len(images)))

    def save_model(self, directory, name):

        if self.vocabulary is None or self.dictionary is None or self.svm is None:

            print("[ERROR] No model. Please create or load one first.")
            return

        # save vocabulary, svm model and dictionary
        vocab_file = path.join(directory, name + "_vocab.pickle")
        with open(vocab_file, 'wb') as outfile:
            pickle.dump(self.vocabulary, outfile)

        svm_file = path.join(directory, name + "_svm.xml")
        self.svm.save(svm_file)

        dictionary_file = path.join(directory, name + "_dict.pickle")
        with open(dictionary_file, 'wb') as outfile:
            pickle.dump(self.dictionary, outfile)

        config_file = path.join(directory, name + "_config.json")
        with open(config_file, 'w') as outfile:
            json.dump(self.config, outfile)

    def best_match(self, imgfile, ratio=0.75, visualize=2, verbose=True):
        """
        :param imgfile: image to be matched against the dataset
        :param ratio: lower values make matching more strict (David Lowe's recommended 0.7 - 0.8)
        :param visualize: 0: no display, 1: display pictures and matches, 2: display (1) + animation
        :param verbose: set to True to display diagnostic information in console
        """
        if self.vocabulary is None or self.dictionary is None or self.svm is None:

            print("[ERROR] Please create or load a model first.")
            return

        # load image, resize it and convert it to grayscale
        image1 = cv2.imread(imgfile)

        if image1.shape[0] > self.resizedH:
            image1 = resize(image1, height=self.resizedH)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # extract keypoints from the image and find best match
        kps1 = self.detector.detect(gray1)
        f = self.BOWextractor.compute(gray1, kps1)
        p = self.svm.predict(f)
        result = p[1][0][0]
        match_name = path.split(self.dictionary[result])[-1].split(".")[0]

        # if the file is not available, just return the label
        if not path.isfile(self.dictionary[result]):
            if verbose:
                print("[INFO] {} file was not found, skipping keypoint matching.".format(self.dictionary[result]))

            # return the name of the matching image and empty object
            return match_name, None

        # else, load the best match and extract its features
        imgfile2 = self.dictionary[result]
        image2 = cv2.imread(imgfile2)

        if image2.shape[0] > self.resizedH:
            image2 = resize(image2, height=self.resizedH)

        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        kps2 = self.detector.detect(gray2)

        # extract features from each of the keypoint regions in the images
        (kps1, features1) = self.extractor.compute(gray1, kps1)
        (kps2, features2) = self.extractor.compute(gray2, kps2)

        # match the keypoints and initialize the list of actual matches
        raw_matches = self.matcher.knnMatch(features1, features2, 2)
        matches = []

        if raw_matches is not None:
            # loop over the raw matches
            for m in raw_matches:
                # ensure the distance passes ratio test
                if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                    matches.append((m[0].trainIdx, m[0].queryIdx))

            n_matches = len(matches)

            if n_matches > 0:
                if verbose:
                    # show some diagnostic information
                    print("[INFO] # of matched keypoints: {}".format(n_matches))

                # draw matches if indicated
                if visualize > 0:
                    self.draw_matches(image1, image2, kps1, kps2, matches, match_name, animation=visualize-1)

                # return the name of the matching image and the number of matches
                return match_name, n_matches

        if verbose:
            print("[INFO] No matches.")
            return None

    def diagnose(self, sample, directory, types=("png", "jpg", "jpeg", "tif"), r_range=(0.5, 0.7, 0.9)):
        # match sample image with each of the images in a directory and
        # present diagnostic information
        s_kps, s_features = self.process_img(sample, verbose=False)
        s_name = path.split(sample)[-1].split(".")[0]
        images = []
        for t in types:
            files = glob(path.join(directory,  "*." + t))
            images.extend(files)

        print("# of images to compare with {}: {}".format(s_name, len(images)))

        # get a list of features and kps for each image
        for imgfile in images:

            name = path.split(imgfile)[-1].split(".")[0]
            print("VS {}:".format(name))
            kps, features = self.process_img(imgfile, verbose=False)
            raw_matches = self.matcher.knnMatch(s_features, features, 2)

            if raw_matches is not None:

                for r in r_range:
                    matches = []
                    # loop over the raw matches
                    for m in raw_matches:
                        # ensure the distance passes ratio test
                        if len(m) == 2 and m[0].distance < m[1].distance * r:
                            matches.append((m[0].trainIdx, m[0].queryIdx))

                    n_matches = len(matches)
                    print("# of matches for ratio = {}: {}".format(r,n_matches))

            else:
                print("No matches were found.")

    def draw_matches(self, image1, image2, kps1, kps2, matches, match_name, animation=1):

        # initialize the output visualization image
        (hA, wA) = image1.shape[:2]
        (hB, wB) = image2.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image1
        vis[0:hB, wA:] = image2
        cv2.namedWindow("Matches")
        # loop over the matches
        for c, (trainIdx, queryIdx) in enumerate(matches):
            # generate color and draw the match
            color = np.random.randint(0, high=255, size=(3,))
            color = tuple(map(int, color))

            pt1 = (int(kps1[queryIdx].pt[0]), int(kps1[queryIdx].pt[1]))
            pt2 = (int(kps2[trainIdx].pt[0] + wA), int(kps2[trainIdx].pt[1]))

            cv2.circle(vis, pt1, 3, color, 2)
            cv2.circle(vis, pt2, 3, color, 2)
            cv2.line(vis, pt1, pt2, color, 1)
            if animation == 1:
                cv2.imshow("Matches", resize(vis, height=600))
                cv2.waitKey(10)

        # display and wait for a key press
        cv2.rectangle(vis, (wA, 0), (wA+wB, 40), (0, 0, 0), -1)
        cv2.putText(vis, match_name, (wA + 10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.imshow("Matches", resize(vis, height=600))
        cv2.waitKey(0)
        cv2.destroyWindow("Matches")

    def train_BOW(self, imagelist, verbose=True):
        # process every image and add every feature
        for imgfile in imagelist:
            kps, features = self.process_img(imgfile, verbose=verbose)
            if len(kps) == 0:
                continue

            features = np.float32(features)
            self.BOWtrainer.add(features)

        # construct vocabulary and apply it to extractor
        if verbose:
            print("[INFO] Training... this may take a while.")

        self.vocabulary = self.BOWtrainer.cluster()
        self.BOWextractor.setVocabulary(self.vocabulary)

    def train_SVM(self, traindata, trainlabels):
        # train svm predictive model
        svm = cv2.ml.SVM_create()
        svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

        # update model
        self.svm = svm


class HistFeatures:

    def __init__(self, histtype="HSV", bins=32, n_neighbors=1, model_dir="", hog_params=None,
                 model_name=None, verbose=True):

        # get parameters
        if model_name is not None:
            config_file = path.join(model_dir, model_name+"_config.json")
            try:
                with open(config_file, "r") as infile:
                    self.config = json.load(infile)
            except FileNotFoundError:
                print("[ERROR] Model configuration file not found.")
                model_name = None

        if model_name is None:
            self.config = {"type": histtype.upper(),
                           "bins": bins,
                           "n_neighbors": n_neighbors,
                           "hog_params": hog_params}

        # histogram parameters
        if self.config["type"] == "RGB":
            self.channels = [0, 1, 2]
            self.mode = "RGB"

        elif self.config["type"] == "GRAY":
            self.channels = [0]
            self.mode = "GRAYSCALE"

        elif self.config["type"] == "HOG":
            self.mode = "HOG"

        else:
            if self.config["type"] != "HSV":
                print("[ERROR] unknown color space. Loading default: HSV")

            self.channels = [0, 1, 2]
            self.mode = "HSV"

        if self.mode != "HOG":
            self.bins = [self.config["bins"]] * len(self.channels)
            self.ranges = [0, 256] * len(self.channels)
            self.hogparams = None
        else:
            self.bins = None
            self.ranges = None
            if self.config["hog_params"] is not None:
                self.hogparams = self.config["hog_params"]
            else:
                if verbose:
                    print("[INFO] loading default HOG parameters")
                self.hogparams = {"imsize": (100, 100), "ori": 9, "ppc": (8, 8), "cpb": (2, 2),
                                  "trans_sqrt": True, "norm": "L1"}

        # model parameters
        self.n_neighbors = self.config["n_neighbors"]
        self.model = None
        self.dictionary = None

        # get objects from file if indicated
        if model_name is not None:
            self.load_model(model_dir, model_name, verbose)

    def calculate_hist(self, imagepath, verbose=True):

        # load image from path and calculate histogram
        image = cv2.imread(imagepath)

        if self.mode == "GRAYSCALE" or self.mode == "HOG":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.mode == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if verbose:
            print("[INFO] Extracting {} histogram from {}".format(self.mode, path.basename(imagepath)))

        if self.mode == "HOG":
            p = self.hogparams
            image = cv2.resize(image, p["imsize"])
            hist = feature.hog(image, orientations=p["ori"], pixels_per_cell=p["ppc"],
                               cells_per_block=p["cpb"], transform_sqrt=p["trans_sqrt"], block_norm=p["norm"])

        else:
            # compute a 3D histogram in the RGB colorspace,
            hist = cv2.calcHist([image], self.channels, None, self.bins, self.ranges)

            # normalize the histogram and convert to a flattened array
            hist = cv2.normalize(hist, hist).flatten()

        return hist

    def process_classes(self, directory, name="hist", types=("png", "jpg", "jpeg", "tif"),
                        autosave=False, verbose=True):
        """
        Generate a model from a dataset of image categories.
        Each subfolder in {directory} represents a class and must contain images that fall into that class.
        The subdirectories' names will be used as labels for the classifier.
        """

        # get a list of subdirectories
        subfolders = glob(path.join(directory,  "*/"))

        dictionary = {}
        features, labels = [], []
        count = 0
        for n, folder in enumerate(subfolders):
            # relate labels to class names and store that information in the dictionary
            dictionary[n] = path.basename(path.dirname(folder))

            # get every image in the subfolder
            for t in types:
                images = glob(path.join(folder, "*." + t))
                for img in images:
                    count += 1
                    features.append(self.calculate_hist(img, verbose=verbose))
                    labels.append(n)

        if verbose:
            print("[INFO] Training model from {} images classified in {} classes... "
                  "this may take a while.".format(count, len(dictionary)))

        # train knn model
        self.dictionary = dictionary
        self.model, acc = self.train_knn(features, labels)

        if verbose:
            print("[INFO] Model created. Accuracy: {:.2f}%".format(acc * 100))

        if autosave:
            self.save_model(directory, name, verbose=verbose)

    def predict(self, imagepath, verbose=True):

        if self.dictionary is None or self.model is None:

            print("[ERROR] Please create or load a model first.")
            return

        # extract features from target image
        hist = self.calculate_hist(imagepath, verbose=verbose)

        # match using knn model
        prediction = self.model.predict(np.array(hist).reshape(1, -1))[0]
        proba = self.model.predict_proba(np.array(hist).reshape(1, -1))[0]

        match_name = self.dictionary[prediction]

        return match_name, max(proba)*100

    def train_knn(self, features, labels):
        # split train and test data
        (trainX, testX, trainY, testY) = train_test_split(np.array(features), np.array(labels), test_size=0.2)

        # generate KNN model
        model = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
        model.fit(trainX, trainY)
        acc = model.score(testX, testY)

        # return trained model and accuracy
        return model, acc

    def save_model(self, directory, name, verbose=True):

        if self.model is None or self.dictionary is None:
            print("[ERROR] No model. Please create or load one first.")
            return

        model_file = path.join(directory, name + "_model.pickle")
        with open(model_file, "wb") as outfile:
            pickle.dump(self.model, outfile)

        dictionary_file = path.join(directory, name + "_dict.pickle")
        with open(dictionary_file, "wb") as outfile:
            pickle.dump(self.dictionary, outfile)

        config_file = path.join(directory, name + "_config.json")
        with open(config_file, 'w') as outfile:
            json.dump(self.config, outfile)

        if verbose:
            print("[INFO] {} model has been saved in classes directory.".format(name))

    def load_model(self, directory, name, verbose=True):

        dictionary_file = path.join(directory, name + "_dict.pickle")
        self.dictionary = load_data(dictionary_file)

        model_file = path.join(directory, name + "_model.pickle")
        self.dictionary = load_data(model_file)

        if self.dictionary is not None and self.model is not None:
            if verbose:
                print("[INFO] Model successfully loaded.")

        else:
            print("[ERROR] Model loading failed. Please load one using "
                  ".load_model(directory, name) or train a new one.")
            self.dictionary, self.model = None, None

