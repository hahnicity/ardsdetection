import json
from random import randint

from flask import Flask, request
from numpy.random import permutation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC

from collate import collate_all_from_breath_meta_to_data_frame
from learn import preprocess_x_y
from sms import send_text

TEST_FRACTION = 0.02

app = Flask(__name__)
# Declare global svm with optimal params
clf = SVC(cache_size=1024, C=10, gamma=0.02)


def get_data():
    df = collate_all_from_breath_meta_to_data_frame(20, None)
    x, y, vents_and_files = preprocess_x_y(df)
    # Reindex to ensure we don't bias the results
    x = x.reindex(permutation(x.index))
    y = y.loc[x.index]
    print("{} positive samples".format(len(y[y['y'] == 1])))
    print("{} negative samples".format(len(y[y['y'] == -1])))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=randint(0, 100)
    )
    return x_train, x_test, y_train, y_test


def train(x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train['y'].values)
    predictions = clf.predict(x_test)
    print("Accuracy: " + str(accuracy_score(y_test['y'], predictions)))
    print("Precision: " + str(precision_score(y_test['y'], predictions)))
    print("Recall: " + str(recall_score(y_test['y'], predictions)))
    fpr, tpr, thresh = roc_curve(y_test['y'], predictions)
    print("False pos rate: " + str(fpr[1]))
    print("True post rate: " + str(tpr[1]))


@app.route('/analyze/', methods=["POST"])
def analyze():
	data = json.loads(request.data)
	breath_data = data["breath_data"]
	patient_id = data["patient_id"]
	print("Received data: " + request.data)
    # Theres a bit of a problem here; because the numbers won't come in scaled
    # to us we may have difficulty actually performing predictions properly.
    # So inevitably when I do a generalization of this work I will need to figure
    # out how to get around this problem.
    #
    # But until then I will not do anything
	prediction = clf.predict([breath_data])
	print("Prediction was: ", prediction)
	if prediction[0] == 1:
		send_text("+19083274527", "+15102543918", "Patient {} has ARDS".format(patient_id))
	return str(prediction[0])


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	train(x_train, x_test, y_train, y_test)
	app.run(debug=True)
