import os
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


pickle_path_model = os.path.join(os.path.dirname(__file__), 'LR_model.pkl')
pickle_path_vectr = os.path.join(os.path.dirname(__file__), 'vctr.pkl')
LR_from_joblib = joblib.load(pickle_path_model)
vectr=TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open(pickle_path_vectr, "rb")))

LR_from_joblib.predict(vectr.fit_transform(['Happy']))
