"""Preprocessing + prediction logic that mirrors smartcart.ipynb exactly.

The saved model.joblib only contains the KMeans model (trained on 3 PCA components),
so the scaler / PCA / one-hot encoder from the notebook are rebuilt here from the
original CSV (same steps, same order) and cached by the app.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = "smartcart_customers.csv"
MODEL_PATH = "model.joblib"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CAT_COLS = ["Education", "Living_With"]

# Columns the app / a batch CSV must provide (ID is NOT needed)
REQUIRED_COLUMNS = [
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", *SPEND_COLS,
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "Complain", "Response",
]

EDU_MAP = {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate",
           "Master": "Postgraduate", "PhD": "Postgraduate"}
LIVING_MAP = {"Married": "Partner", "Together": "Partner", "Single": "Alone",
              "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"}

# Segment names describe the saved model.joblib (derived from the cluster averages).
# If you retrain the model, re-check these.
SEGMENTS = {
    0: ("Budget Families", "Low income, low spending, partnered, about 1 child at home. Heavy web browsing, few purchases, rarely responds to campaigns."),
    1: ("Deal-Seeking Regulars", "Upper-middle income, solid spending, older. Many web, catalog and discounted purchases."),
    2: ("Premium Big Spenders", "Highest income and spending, few children. Heavy catalog and store buyers with the best campaign response."),
    3: ("Budget Singles", "Low income, low spending, all living alone, about 1 child at home. Browse the web a lot, moderate campaign response."),
}


def _engineer(df: pd.DataFrame, ref_date: pd.Timestamp, income_median: float) -> pd.DataFrame:
    """Notebook feature engineering. Returns the frame BEFORE outlier removal / encoding."""
    d = df.copy()
    d["Income"] = d["Income"].fillna(income_median)
    d["Age"] = 2026 - d["Year_Birth"]                     # notebook uses the fixed year 2026
    dt = pd.to_datetime(d["Dt_Customer"], dayfirst=True)
    d["Customer_Tenure_Days"] = (ref_date - dt).dt.days
    d["Total_Spending"] = d[SPEND_COLS].sum(axis=1)
    d["Total_Children"] = d["Kidhome"] + d["Teenhome"]
    d["Education"] = d["Education"].replace(EDU_MAP)
    d["Living_With"] = d["Marital_Status"].replace(LIVING_MAP)
    drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", *SPEND_COLS]
    return d.drop(columns=[c for c in drop if c in d.columns])


class Segmenter:
    def __init__(self, csv_path: str = CSV_PATH, model_path: str = MODEL_PATH):
        raw = pd.read_csv(csv_path)
        self.income_median = raw["Income"].median()
        self.ref_date = pd.to_datetime(raw["Dt_Customer"], dayfirst=True).max()

        clean = _engineer(raw, self.ref_date, self.income_median)
        clean = clean[(clean["Age"] < 90) & (clean["Income"] < 600_000)]   # outliers, as in notebook

        self.ohe = OneHotEncoder()
        enc = pd.DataFrame(self.ohe.fit_transform(clean[CAT_COLS]).toarray(),
                           columns=self.ohe.get_feature_names_out(CAT_COLS), index=clean.index)
        X = pd.concat([clean.drop(columns=CAT_COLS), enc], axis=1)
        self.feature_order = list(X.columns)

        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=3).fit(self.scaler.transform(X))
        self.model = joblib.load(model_path)               # your KMeans(n_clusters=4)

        # profile of each segment on the training data (for display)
        X_prof = X.copy()
        X_prof["Segment"] = self.model.predict(self.pca.transform(self.scaler.transform(X)))
        self.profile = X_prof.groupby("Segment").mean().round(2)
        self.sizes = X_prof["Segment"].value_counts().sort_index()

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = _engineer(df[REQUIRED_COLUMNS], self.ref_date, self.income_median)
        enc = pd.DataFrame(self.ohe.transform(clean[CAT_COLS]).toarray(),
                           columns=self.ohe.get_feature_names_out(CAT_COLS), index=clean.index)
        X = pd.concat([clean.drop(columns=CAT_COLS), enc], axis=1)
        return X[self.feature_order]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.features(df)
        return self.model.predict(self.pca.transform(self.scaler.transform(X)))
