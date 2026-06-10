from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

def get_outlier_detector():
    # Tailored to clean up leverage points for parametric/distance estimation matrices
    return IsolationForest(contamination=0.07, random_state=42)

def get_parametric_pipelines():
    # These architectures assume symmetric distributions or rely strictly on geometric metrics
    return {
        "Logistic Regression": Pipeline([
            ("scale", PowerTransformer(method='yeo-johnson')),
            ("model", LogisticRegression())
        ]),
        "SGD": Pipeline([
            ("scale", PowerTransformer(method='yeo-johnson')),
            ("model", SGDClassifier(eta0=0.0005, max_iter=2300, tol=0.005, random_state=42))
        ]),
        "SVC": Pipeline([
            ("scale", PowerTransformer(method='yeo-johnson')),
            ("model", SVC(random_state=42))
        ]),
        "KNN": Pipeline([
            ("scale", PowerTransformer(method='yeo-johnson')),
            ("model", KNeighborsClassifier())
        ]),
        "Naive Bayes": Pipeline([
            ("scale", PowerTransformer(method='yeo-johnson')),
            ("model", GaussianNB())
        ])
    }

def get_non_parametric_models():
    # Tree frameworks are structurally invariant to monotonic scaling transformations and extreme outliers
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }