import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout and hide Streamlit menu/footer for clean look
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .block-container {
                background: rgba(255,255,255,0.92) !important;
                border-radius: 16px;
                padding: 2rem !important;
                margin-bottom: 2rem !important;
                box-shadow: 0 4px 24px rgba(0,0,0,0.04);
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add a soft, businesslike background image (online source, no copyright issues)
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Flight Price Prediction")
page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Data & EDA",
        "Feature Engineering",
        "Model Building & Results",
        "Insights & Recommendations",
        "About Project"
    ]
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Flight_Booking.csv")
    return df

df = load_data()

# Main Sections
if page == "Project Overview":
    st.title("✈️ Flight Price Prediction")
    st.markdown("""
    <div class='block-container'>
    <h3>Business Problem</h3>
    <ul>
        <li><b>Objective:</b> Predict flight prices using advanced regression models and understand the drivers behind price variation.</li>
        <li><b>Impact:</b> Better price forecasting helps airlines, agencies, and platforms optimize revenue and offer competitive prices.</li>
        <li><b>Approach:</b> Data analysis, feature engineering, and supervised ML (OLS Regression & Random Forest).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='block-container'>
    <h4>Project Steps</h4>
    <ol>
        <li>Exploratory Data Analysis (EDA)</li>
        <li>Feature Engineering & Preprocessing</li>
        <li>Model Building & Evaluation</li>
        <li>Business Insights & Recommendations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

elif page == "Data & EDA":
    st.header("📊 Data Exploration & EDA")

    st.markdown("""
    <div class='block-container'>
    <b>Quick Data Overview:</b> The dataset contains 300,000+ bookings with columns like airline, flight, city, class, stops, duration, days left to departure, and price.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # 1. Price Distribution
    st.subheader("Distribution of Flight Prices")
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    sns.histplot(df['price'], bins=40, kde=True, ax=ax1)
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Histogram of Flight Prices")
    st.pyplot(fig1)
    st.info("Most flight prices are under ₹15,000, but there are some high outliers. This suggests most tickets are in the affordable range, but a few premium tickets skew the distribution.")

    # 2. Airlines Market Share
    st.subheader("Airlines Market Share")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    airline_counts = df['airline'].value_counts().head(10)
    sns.barplot(x=airline_counts.index, y=airline_counts.values, ax=ax2)
    ax2.set_xlabel("Airline")
    ax2.set_ylabel("Number of Bookings")
    ax2.set_title("Top 10 Airlines by Booking Volume")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    st.info("Top airlines dominate booking volume, suggesting strong brand loyalty or aggressive pricing.")

    # 3. Price by Source City
    st.subheader("Average Price by Source City")
    city_price = df.groupby('source_city')['price'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8,3))
    sns.barplot(x=city_price.index, y=city_price.values, ax=ax3)
    ax3.set_xlabel("Source City")
    ax3.set_ylabel("Average Price")
    ax3.set_title("Mean Flight Price by Source City")
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.info("Certain source cities (often metros or high-demand hubs) command higher average prices.")

    # 4. Price by Class
    st.subheader("Flight Price by Class")
    fig4, ax4 = plt.subplots(figsize=(5,3))
    sns.boxplot(x='class', y='price', data=df, ax=ax4)
    ax4.set_title("Price Distribution by Class")
    st.pyplot(fig4)
    st.info("Business class fares are significantly higher than economy class, as expected.")

    # 5. Stops vs Price
    st.subheader("Stops vs. Price")
    fig5, ax5 = plt.subplots(figsize=(6,3))
    sns.boxplot(x='stops', y='price', data=df, ax=ax5)
    ax5.set_title("Price vs. Stops")
    st.pyplot(fig5)
    st.info("Non-stop flights are typically more expensive. Flights with more stops generally cost less, possibly due to inconvenience.")

    # 6. Days Left vs Price
    st.subheader("Days Left to Departure vs Price")
    fig6, ax6 = plt.subplots(figsize=(8,3))
    sns.lineplot(x='days_left', y='price', data=df.sample(5000), alpha=0.4, marker="o", lw=0)
    sns.regplot(x='days_left', y='price', data=df.sample(5000), scatter=False, ax=ax6, color="red")
    ax6.set_title("Ticket Price as Departure Approaches")
    st.pyplot(fig6)
    st.info("Prices generally increase as the departure date approaches. Booking early can secure lower fares.")

    # 7. Duration vs Price
    st.subheader("Flight Duration vs Price")
    fig7, ax7 = plt.subplots(figsize=(8,3))
    sns.scatterplot(x='duration', y='price', data=df.sample(5000), alpha=0.3)
    ax7.set_xlabel("Flight Duration (hours)")
    ax7.set_ylabel("Price")
    ax7.set_title("Flight Duration vs. Price")
    st.pyplot(fig7)
    st.info("Longer flights tend to cost more, but there is wide variability. Some short flights are premium-priced, likely due to business demand or limited routes.")

    # Add any other plots from your notebook here...

elif page == "Feature Engineering":
    st.header("🛠️ Feature Engineering & Data Preparation")

    st.markdown("""
    <div class='block-container'>
    <ul>
        <li>Missing value treatment: None required (no missing values found).</li>
        <li>Label encoding applied to categorical features (airline, source, destination, class, etc).</li>
        <li>Outlier handling done for the price variable, but retaining outliers actually improved model performance.</li>
        <li>Variance Inflation Factor (VIF) checked to reduce multicollinearity in the final features set.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("Key categorical variables were encoded and features with high VIF were removed to boost model performance and stability.")

elif page == "Model Building & Results":
    st.header("🤖 Model Building & Evaluation")

    st.markdown("""
    <div class='block-container'>
    <b>Models Compared:</b>
    <ul>
        <li><b>OLS Linear Regression:</b> On full data and on data with outliers removed</li>
        <li><b>Random Forest Regressor:</b> On preprocessed data</li>
    </ul>
    <b>Evaluation Metrics:</b> MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
    </div>
    """, unsafe_allow_html=True)

    results = pd.DataFrame({
        "Model": ["OLS (Base)", "OLS (No Outliers)", "Random Forest"],
        "MAE": [4626.1, 4659.04, 1109.83],
        "RMSE": [7005.1, 7026.8, 2783.05]
    })
    st.subheader("Model Performance Comparison")
    st.table(results)

    # Add performance bar plot
    st.subheader("Visual Comparison: Model Errors")
    fig_perf, ax_perf = plt.subplots(figsize=(6,3))
    index = np.arange(len(results))
    bar1 = ax_perf.bar(index-0.15, results["MAE"], 0.3, label='MAE')
    bar2 = ax_perf.bar(index+0.15, results["RMSE"], 0.3, label='RMSE')
    ax_perf.set_xticks(index)
    ax_perf.set_xticklabels(results["Model"])
    ax_perf.set_ylabel("Error Value")
    ax_perf.set_title("MAE & RMSE by Model")
    ax_perf.legend()
    st.pyplot(fig_perf)

    st.info("""
    The Random Forest model delivers far lower MAE and RMSE.  
    For business, this means far more accurate price prediction and fewer lost sales due to price misestimation.
    """)

    st.markdown("""
    <div class='block-container'>
    <b>Key Takeaways:</b>
    <ul>
        <li>Base OLS performed better than OLS with outliers removed.</li>
        <li>Random Forest outperformed all linear models by a large margin.</li>
        <li>Retaining outliers sometimes helps capture the real pricing extremes in airline tickets.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Insights & Recommendations":
    st.header("💡 Business Insights & Recommendations")

    st.markdown("""
    <div class='block-container'>
    <b>Major Business Insights:</b>
    <ul>
        <li><b>Dynamic Pricing:</b> Ticket prices rise as the departure date approaches. Early-bird discounts could drive sales.</li>
        <li><b>Class & Stops:</b> Business class is a high-margin segment; non-stop flights can be premium priced.</li>
        <li><b>Key Drivers:</b> Airline, route, booking timing, and class have the biggest influence on prices.</li>
        <li><b>Prediction Power:</b> Random Forest enables precise price suggestions for customers, reducing manual overrides.</li>
    </ul>
    <b>Recommendations:</b>
    <ul>
        <li>Implement Random Forest-based price prediction for maximum accuracy.</li>
        <li>Use OLS linear models for interpretability when explaining pricing to business stakeholders.</li>
        <li>Continuously monitor real-world model performance as the market evolves.</li>
        <li>Offer special deals based on booking timing and segment (class, stops).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "About Project":
    st.header("ℹ️ About Project & Data")
    st.markdown("""
    <div class='block-container'>
    <p>
    <b>Dataset:</b> 300,000+ flight bookings with key info on airline, city, route, class, and price.<br>
    <b>Project Lead:</b> [Madhusudhanan]<br>
    <b>Use Case:</b> Business-ready, AI-powered flight price prediction.<br>
    <b>Tech Stack:</b> Python, pandas, matplotlib, seaborn, scikit-learn, Streamlit.<br>
    </p>
    </div>
    """, unsafe_allow_html=True)
