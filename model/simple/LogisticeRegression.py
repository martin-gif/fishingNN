from sklearn.linear_model import LogisticRegression


def get_LogisticRegression_model():
    model = LogisticRegression(penalty="l2")
    return model
