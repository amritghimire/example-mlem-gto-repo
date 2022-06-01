from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from mlem.api import save


def main():
    data, y = load_iris(return_X_y=True, as_frame=True)
    rf = RandomForestClassifier(
        n_jobs=2,
        random_state=42,
    )
    rf.fit(data, y)

    rf_2 = RandomForestClassifier(
        n_jobs=2,
        random_state=4,
    )
    rf_2.fit(data, y)


    save(
        rf,
        "rf",
        sample_data=data,
        description="Random Forest Classifier",
    )
    save(
        rf_2,
        "rf_state_4",
        sample_data=data,
        description="Random Forest Classifier v2",
    )



if __name__ == "__main__":
    main()
