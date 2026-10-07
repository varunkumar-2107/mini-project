import datetime as dt

import pandas as pd
import streamlit as st

from segmentation import REQUIRED_COLUMNS, SEGMENTS, Segmenter

st.set_page_config(page_title="SmartCart Customer Segments", page_icon="🛒", layout="wide")


@st.cache_resource
def get_segmenter() -> Segmenter:
    return Segmenter()


seg = get_segmenter()

st.title("🛒 SmartCart – Customer Segmentation")
st.caption("Enter a customer's details to find out which of the 4 customer segments they belong to "
           "(KMeans model).")

tab_one, tab_batch, tab_info = st.tabs(["Single customer", "Batch (CSV upload)", "About the segments"])

# --------------------------------------------------------------- single customer
with tab_one:
    with st.form("customer"):
        st.subheader("Profile")
        c1, c2, c3 = st.columns(3)
        year_birth = c1.number_input("Year of birth", 1937, 2005, 1970)
        education = c2.selectbox("Education", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
        marital = c3.selectbox("Marital status", ["Married", "Together", "Single", "Divorced", "Widow"])

        c1, c2, c3, c4 = st.columns(4)
        income = c1.number_input("Yearly income", 0.0, 600000.0, 50000.0, step=1000.0)
        kidhome = c2.number_input("Kids at home", 0, 5, 0)
        teenhome = c3.number_input("Teens at home", 0, 5, 0)
        dt_customer = c4.date_input("Customer since", dt.date(2013, 6, 1),
                                    min_value=dt.date(2012, 1, 1), max_value=seg.ref_date.date())

        st.subheader("Spending in the last 2 years")
        c1, c2, c3 = st.columns(3)
        wines = c1.number_input("Wines", 0, 5000, 300)
        fruits = c2.number_input("Fruits", 0, 1000, 25)
        meat = c3.number_input("Meat products", 0, 5000, 150)
        c1, c2, c3 = st.columns(3)
        fish = c1.number_input("Fish products", 0, 1000, 35)
        sweets = c2.number_input("Sweet products", 0, 1000, 25)
        gold = c3.number_input("Gold products", 0, 1000, 40)

        st.subheader("Behaviour")
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Days since last purchase", 0, 365, 50)
        deals = c2.number_input("Purchases with discount", 0, 30, 2)
        visits = c3.number_input("Web visits last month", 0, 30, 5)
        c1, c2, c3 = st.columns(3)
        web_p = c1.number_input("Web purchases", 0, 50, 4)
        cat_p = c2.number_input("Catalog purchases", 0, 50, 2)
        store_p = c3.number_input("Store purchases", 0, 50, 5)
        c1, c2 = st.columns(2)
        complain = c1.selectbox("Complained in last 2 years?", ["No", "Yes"])
        response = c2.selectbox("Accepted the last campaign offer?", ["No", "Yes"])

        go = st.form_submit_button("Find segment", type="primary")

    if go:
        row = pd.DataFrame([{
            "Year_Birth": year_birth, "Education": education, "Marital_Status": marital,
            "Income": income, "Kidhome": kidhome, "Teenhome": teenhome,
            "Dt_Customer": dt_customer.strftime("%d-%m-%Y"), "Recency": recency,
            "MntWines": wines, "MntFruits": fruits, "MntMeatProducts": meat,
            "MntFishProducts": fish, "MntSweetProducts": sweets, "MntGoldProds": gold,
            "NumDealsPurchases": deals, "NumWebPurchases": web_p,
            "NumCatalogPurchases": cat_p, "NumStorePurchases": store_p,
            "NumWebVisitsMonth": visits, "Complain": int(complain == "Yes"),
            "Response": int(response == "Yes"),
        }])[REQUIRED_COLUMNS]

        cluster = int(seg.predict(row)[0])
        name, desc = SEGMENTS.get(cluster, (f"Segment {cluster}", ""))
        st.divider()
        st.success(f"### Segment {cluster}: {name}")
        st.write(desc)

        feats = seg.features(row).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age", int(feats["Age"]))
        m2.metric("Total spending", f"{feats['Total_Spending']:,.0f}")
        m3.metric("Children at home", int(feats["Total_Children"]))
        m4.metric("Days as customer", int(feats["Customer_Tenure_Days"]))

        st.write("**Typical customer in this segment vs. this customer**")
        cols = ["Income", "Total_Spending", "Total_Children", "NumWebPurchases",
                "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
        cmp_df = pd.DataFrame({"Segment average": seg.profile.loc[cluster, cols],
                               "This customer": feats[cols]})
        st.dataframe(cmp_df.round(1), use_container_width=True)

# --------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with these columns (`ID` is optional):")
    st.code(", ".join(REQUIRED_COLUMNS))
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        data = pd.read_csv(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = data.copy()
            out["Segment"] = seg.predict(data)
            out["Segment_Name"] = out["Segment"].map(lambda c: SEGMENTS.get(c, (str(c),))[0])
            st.dataframe(out, use_container_width=True)
            st.bar_chart(out["Segment_Name"].value_counts())
            st.download_button("Download results", out.to_csv(index=False).encode(),
                               "customer_segments.csv", "text/csv")

# --------------------------------------------------------------- info
with tab_info:
    st.subheader("Segment profiles (training data averages)")
    prof = seg.profile.copy()
    prof.insert(0, "Name", [SEGMENTS.get(i, (str(i),))[0] for i in prof.index])
    prof.insert(1, "Customers", seg.sizes)
    st.dataframe(prof, use_container_width=True)
    for i, (n, d) in SEGMENTS.items():
        st.markdown(f"**Segment {i} – {n}:** {d}")
