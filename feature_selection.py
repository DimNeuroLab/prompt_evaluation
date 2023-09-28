import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def get_feature_importance(df, method='dt'):
    METHOD_DICT = {'dt': DecisionTreeClassifier, 'rf': RandomForestClassifier}
    assert method in METHOD_DICT.keys()

    x_data = df.loc[:, (df.columns != 'prompt') & (df.columns != 'Good Bad (1,-1)')].values.tolist()
    y_data = df['Good Bad (1,-1)'].to_numpy()

    print(f'Running {method.upper()}')
    clf = METHOD_DICT[method](random_state=0, max_depth=4)
    if method == 'rf':
        clf.set_params(n_estimators=300)
    clf.fit(x_data, y_data)

    pred = clf.predict(x_data)
    print(f'ACC: {accuracy_score(y_data, pred)}')
    print(f'F1: {f1_score(y_data, pred, average="macro")}')

    feature_importance_dict = dict(zip(df.columns[2:], clf.feature_importances_))

    return feature_importance_dict


if __name__ == "__main__":
    DATA = pd.read_csv("data/all_features_annotated.tsv", sep='\t')

    feature_importances = get_feature_importance(DATA, method="rf")
    print(feature_importances)
    print("\n".join(f"{key}: {value}" for key, value in feature_importances.items()))
