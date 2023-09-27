import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def get_feature_importance(df, method='dt'):
    assert method in ['dt', 'rf']

    x_data = df.loc[:, (df.columns != 'prompt') & (df.columns != 'Good Bad (1,-1)')].values.tolist()
    y_data = df['Good Bad (1,-1)'].to_numpy()

    if method == 'dt':
        print('Running Decision Tree')
        clf = DecisionTreeClassifier(random_state=0, max_depth=4)
    else:
        print('Running Random Forest')
        clf = RandomForestClassifier(n_estimators=300, random_state=0, max_depth=4)
    clf.fit(x_data, y_data)

    pred = clf.predict(x_data)
    print('ACC', accuracy_score(y_data, pred))
    print('F1', f1_score(y_data, pred, average='macro'))

    importance_list = list(clf.feature_importances_)
    feature_names = list(df.columns[2:])
    importance_dict = {f_name: importance_list[idx] for idx, f_name in enumerate(feature_names)}
    return importance_dict


if __name__ == '__main__':
    DATA = pd.read_csv('data/all_features_annotated.tsv', sep='\t')

    i_dict = get_feature_importance(DATA, method='rf')
    print(i_dict)

    for k, v in i_dict.items():
        print(k)
    for k, v in i_dict.items():
        print(v)
