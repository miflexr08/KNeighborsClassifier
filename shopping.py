import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# check50 ai50/projects/2024/x/shopping
def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE)

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    with open('shopping.csv', 'r', newline='\n') as ocurrences_csv:
        ocurrences = csv.reader(ocurrences_csv, delimiter=',')

        data_labels : tuple[list[list], [int]] = ([], [])
        for features in ocurrences:
            if features[0] == "Administrative":
                continue

            input = [
                int(features[0]), # Administrative,
                float(features[1]), # Administrative_Duration,
                int(features[2]), # Informational,
                float(features[3]), # Informational_Duration,
                int(features[4]), # ProductRelated,
                float(features[5]), # ProductRelated_Duration,
                float(features[6]), # BounceRates,
                float(features[7]), # ExitRates,
                float(features[8]), # PageValues,
                float(features[9]), # SpecialDay,
                get_month(features[10]), # Month,
                int(features[11]), # OperatingSystems,
                int(features[12]), # Browser,
                int(features[13]), # Region,
                int(features[14]), # TrafficType,
                int(get_visitor(features[15])), # VisitorType,
                1 if features[16] == "TRUE" else 0  # Weekend,
            ]

            data, labels = data_labels
            labels.append(1 if features[17] == "TRUE" else 0)
            data.append(input)


        #print_evidence, print_labels = data_labels
        #print(f"evidence: {print_evidence}")
        #print(f"labels: {print_labels}")

        return data_labels

def get_month(month : str):
    return {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
        "May": 4, "June": 5, "Jul": 6, "Aug": 7,
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11 }.get(month)

def get_visitor(visitor : str):
    return {
        "Returning_Visitor": 1, "New_Visitor": 0, "Other": 0 }.get(visitor)

def train_model(evidence, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)

    print(f"neigh: {neigh}")

    return neigh

def evaluate(labels, predictions):

    assert len(labels) == len(predictions)

    zipped = zip(labels, predictions)
    #print(list(zipped))

    sensitivity = 0
    specificity = 0
    counter = 1
    for label, prediction in list(zipped):
        if label == 1:
            if prediction == 1:
                sensitivity += (sensitivity + 1) / counter
            else:
                sensitivity = sensitivity / counter
        elif label == 0:
            if prediction == 0:
                specificity += (specificity + 1) / counter
            else:
                specificity = specificity / counter
        counter += 1

    return sensitivity, specificity







if __name__ == "__main__":
    main()
