import cv2
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create
from imutils import resize
from glob import glob
from os import path
import pickle
import numpy as np
import json


class KPFeatures:

    def __init__(self, detector="SURF", extractor="SURF", matcher="BruteForce", flann_params=None,
                 dictsize=500, model_dir="", model_name=None, verbose=True):
        """
        :param detector: "BRISK", "DENSE", "FAST", "GFTT", "HARRIS", "MSER", "ORB", "SIFT", "SURF", "STAR"
        :param extractor: "SIFT", "ROOTSIFT", "SURF", "BRIEF", "ORB", "BRISK", "FREAK"
        :param matcher: "FLANN", else "BruteForce" will be used
        """
        # get parameters
        if model_name is not None:
            model_path = path.join(model_dir, model_name+"_config.json")
            try:
                with open(model_path, "r") as infile:
                    self.config = json.load(infile)
            except FileNotFoundError:
                print("[ERROR] Model configuration file not found.")
                model_name = None

        if model_name is None:
            if verbose:
                print("Loading default parameters.")
            self.config = {"detector": detector,
                           "extractor": extractor,
                           "matcher": matcher,
                           "flann_params": flann_params,
                           "dictsize": dictsize}

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
        self.vocabulary = self.load_data(vocab_file)
        self.BOWextractor.setVocabulary(self.vocabulary)

        svm_file = path.join(directory, name + "_svm.xml")
        self.svm = cv2.ml.SVM_load(svm_file)

        dictionary_file = path.join(directory, name + "_dict.pickle")
        self.dictionary = self.load_data(dictionary_file)

        if self.vocabulary is not None and self.dictionary is not None and self.svm is not None:
            if verbose:
                print("Model successfully loaded.")

        else:
            print("[ERROR] Model loading failed. Please load one using "
                  ".load_model(directory, name) or train a new one.")
            self.vocabulary, self.dictionary, self.svm = None, None, None

    def process_img(self, imgfile, verbose=True):

        # load the images and convert to grayscale
        image = cv2.imread(imgfile)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints
        kps = self.detector.detect(gray)

        (kps, features) = self.extractor.compute(gray, kps)

        if verbose:
            print(imgfile, "# of keypoints: {}".format(len(kps)))

        return kps, features

    def process_batch(self, directory, types=("png", "jpg"), name="batch", autosave=False, verbose=True):
        # generate a model from a dataset of images
        imglist = []
        for t in types:
            images = glob(path.join(directory,  "*." + t))
            # process every image and add every feature
            for imgfile in images:
                imglist.append(imgfile)
                kps, features = self.process_img(imgfile, verbose=verbose)
                features = np.float32(features)
                self.BOWtrainer.add(features)

        # construct vocabulary and apply it to extractor to create the scv model
        if verbose:
            print("Training... this may take a while.")

        self.vocabulary = self.BOWtrainer.cluster()
        self.BOWextractor.setVocabulary(self.vocabulary)

        traindata, trainlabels,  = [], []
        dictionary = {}
        for n, im in enumerate(imglist):
            image = cv2.imread(im)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # create train data from BOW vectors
            traindata.extend(self.BOWextractor.compute(gray, self.detector.detect(gray)))
            trainlabels.append(n)
            # relate labels to image paths and store that information in the dictionary
            dictionary[n] = im

        # train svm predictive model
        svm = cv2.ml.SVM_create()
        svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

        # update model
        self.svm = svm
        self.dictionary = dictionary

        if autosave:
            self.save_model(directory, name)

        if verbose:
            print("Model created from database containing {} images.".format(len(imglist)))

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

            print("Please create or load a model first.")
            return

        # load image in grayscale
        image1 = cv2.imread(imgfile)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # extract keypoints from the image and find best match
        kps1 = self.detector.detect(gray1)
        f = self.BOWextractor.compute(gray1, kps1)
        p = self.svm.predict(f)
        result = p[1][0][0]

        # extract keypoints from the best match
        imgfile2 = self.dictionary[result]
        image2 = cv2.imread(imgfile2)
        match_name = path.split(imgfile2)[-1].split(".")[0]
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
                    print("# of matched keypoints: {}".format(n_matches))

                # draw matches if indicated
                if visualize > 0:
                    self.draw_matches(image1, image2, kps1, kps2, matches, match_name, animation=visualize-1)

                # return the name of the matching image
                return match_name, n_matches

        if verbose:
            print("No matches.")
            return None

    def diagnose(self, sample, directory, types=("png", "jpg"), r_range=(0.5, 0.7, 0.9)):
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

    def load_data(self, file):

        try:
            with open(file, 'rb') as infile:
                obj = pickle.load(infile)

            return obj

        except FileNotFoundError:
            print("[ERROR] File not found. {}".format(file))
            return None

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

