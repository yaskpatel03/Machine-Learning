### Imports ###
import sys
from predictor import MultiColumnLabelEncoder, predict_runs


"""
sys.argv[1] is the input test file name given as command line arguments

"""
runs = predict_runs(sys.argv[1])
print("Predicted Runs: ", runs)