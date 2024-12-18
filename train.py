import pandas as pd

# File paths
data_dict_path = "data/data_dictionary.csv"
train_path = "data/train.csv"
test_path = "data/test.csv"
sample_submission_path = "data/sample_submission.csv"

# Load the data
data_dict = pd.read_csv(data_dict_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Display the first few rows of each file
data_dict_head = data_dict.head()
train_data_head = train_data.head()
test_data_head = test_data.head()
sample_submission_head = sample_submission.head()

data_dict_head, train_data_head, test_data_head, sample_submission_head
