import os
import streamlit as st
from databricks import sql
import pandas as pd

st.set_page_config(page_title="UC Access Control Demo")
st.title("Iteration Filter Demo")

host = os.environ["DATABRICKS_HOST"].replace("https://", "")
http_path = os.environ["DATABRICKS_SQL_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]

try:
    with sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    ) as connection:
        cursor = connection.cursor()
        # Show current user
        cursor.execute("SELECT current_user()")
        current_user = cursor.fetchall()[0][0]
        st.success(f"Logged in as: {current_user}")
        # Query secure view
        cursor.execute("""
            SELECT *
            FROM tiger_assignment.tiger_bronze.vw_sales_secure
        """)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        st.subheader("Data Returned")
        if "country" in df.columns:
            selected_country = st.selectbox(
                "Select Country",
                ["All"] + sorted(df["country"].unique().tolist())
            )
            if selected_country != "All":
                df = df[df["country"] == selected_country]
        st.metric("Rows Visible", len(df))
        st.dataframe(df)
        if (
            "country" in df.columns and
            "sales_amount" in df.columns and
            not df.empty
        ):
            chart_df = (
                df.groupby("country")["sales_amount"]
                .sum()
                .reset_index()
            )
            st.subheader("Revenue by Country")
            st.bar_chart(
                chart_df.set_index("country")
            )
except Exception as e:
    st.error(f"Error: {str(e)}")