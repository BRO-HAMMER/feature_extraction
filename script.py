from BROFEATURES import KPFeatures

FE = KPFeatures()
FE.predict("dr2.png", datasetdir="dataset", modelname="batch", from_files=True, threshold=0.5)