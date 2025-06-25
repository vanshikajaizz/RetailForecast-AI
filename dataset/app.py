# ✅ Essential Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()
import io




import os
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

def send_email_alert(subject, body, to_email):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        print("📧 Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")




# 🌐 Language Selection
language = st.sidebar.selectbox("🌐 Select Language", ["English", "Hindi","Spanish","French"])


translations = {
    "English": {
        "title": "📈 Walmart Demand Forecasting - RetailForecast AI",
        "subtitle": "Get weekly sales predictions and explore real-time trends with interactive analysis.",
        "filter_header": "🔍 Filter for Prediction",
        "select_store": "Select Store",
        "select_dept": "Select Department",
        "select_date": "Select Date",
        "high_alert": "⚠️ High Sales Alert! Monitor stock and logistics.",
        "low_alert": "📉 Low Sales Alert! Investigate potential demand issues.",
        "predicted": "📊 Predicted Weekly Sales",
        "actual": "📌 Actual Sales",
    },
    "Hindi": {
        "title": "📈 वॉलमार्ट मांग पूर्वानुमान - रिटेलफोरकास्ट एआई",
        "subtitle": "साप्ताहिक बिक्री की भविष्यवाणी प्राप्त करें और वास्तविक समय के रुझान का विश्लेषण करें।",
        "filter_header": "🔍 भविष्यवाणी के लिए फ़िल्टर",
        "select_store": "स्टोर चुनें",
        "select_dept": "विभाग चुनें",
        "select_date": "तारीख चुनें",
        "high_alert": "⚠️ अधिक बिक्री की चेतावनी! स्टॉक और लॉजिस्टिक्स की निगरानी करें।",
        "low_alert": "📉 कम बिक्री की चेतावनी! संभावित समस्याओं की जांच करें।",
        "predicted": "📊 अनुमानित साप्ताहिक बिक्री",
        "actual": "📌 वास्तविक बिक्री",
    },
    "Spanish": {
        "title": "📈 Predicción de Ventas de Walmart - RetailForecast AI",
        "subtitle": "Obtén predicciones semanales de ventas y explora tendencias en tiempo real.",
        "filter_header": "🔍 Filtro para Predicción",
        "select_store": "Seleccionar Tienda",
        "select_dept": "Seleccionar Departamento",
        "select_date": "Seleccionar Fecha",
        "high_alert": "⚠️ ¡Alerta de Ventas Altas! Supervisa el inventario y la logística.",
        "low_alert": "📉 ¡Alerta de Ventas Bajas! Investiga posibles problemas.",
        "predicted": "📊 Ventas Semanales Previstos",
        "actual": "📌 Ventas Reales",
    },
    "French": {
        "title": "📈 Prévision des Ventes Walmart - RetailForecast AI",
        "subtitle": "Obtenez des prévisions hebdomadaires et explorez les tendances en temps réel.",
        "filter_header": "🔍 Filtre pour la Prédiction",
        "select_store": "Sélectionner un Magasin",
        "select_dept": "Sélectionner un Département",
        "select_date": "Sélectionner une Date",
        "high_alert": "⚠️ Alerte de Forte Vente ! Surveillez le stock et la logistique.",
        "low_alert": "📉 Alerte de Faible Vente ! Vérifiez les problèmes potentiels.",
        "predicted": "📊 Ventes Hebdomadaires Prévues",
        "actual": "📌 Ventes Réelles",
    }
}





# ✅ Page Configuration
st.set_page_config(page_title="RetailForecast AI", layout="wide")

# Global Styling
st.markdown("""
    <style>
    /* Font & layout */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        color: #ffffff;
    }

    /* Metric boxes */
    div[data-testid="metric-container"] {
        background: #232323;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 0 10px #444;
        margin: 10px 0;
    }

    /* Expander panel */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 16px;
    }

    .stApp {
        background-color: rgba(0, 0, 0, 0.6); /* fallback bg in case bg.jpg not visible */
    }

    </style>
""", unsafe_allow_html=True)


# ✅ Add Background Image
def add_bg(img_file):
    with open(img_file, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg("assets/bg.jpg")



# ✅ Load Model and Data
model = joblib.load("model/model.pkl")
final_df = pd.read_csv("final_df.csv")
final_df['Date'] = pd.to_datetime(final_df['Date'])
final_df['Year'] = final_df['Date'].dt.year

# ✅ Clean up IsHoliday Columns if Needed
# if 'IsHoliday_x' in final_df.columns and 'IsHoliday_y' in final_df.columns:
#     if (final_df['IsHoliday_x'] == final_df['IsHoliday_y']).all():
#         final_df.drop(columns=['IsHoliday_x'], inplace=True)
#         final_df.rename(columns={'IsHoliday_y': 'IsHoliday'}, inplace=True)
#     else:
#         final_df.rename(columns={'IsHoliday_y': 'IsHoliday'}, inplace=True)

# # ✅ App Title
# st.title("📈 Walmart Demand Forecasting - RetailForecast AI")
# st.markdown("Get weekly sales predictions and explore real-time trends with interactive analysis.")

# # ✅ Sidebar Filters
# st.sidebar.header("🔍 Filter for Prediction")
# store = st.sidebar.selectbox("Select Store", sorted(final_df['Store'].unique()))
# dept = st.sidebar.selectbox("Select Department", sorted(final_df['Dept'].unique()))
# date = st.sidebar.date_input("Select Date", final_df['Date'].min())




# # ✅ Prediction Logic
# selected = final_df[(final_df['Store'] == store) & (final_df['Dept'] == dept) & (final_df['Date'] == pd.to_datetime(date))]

# if selected.empty:
#     st.warning("No data found for this combination.")
# else:
#     features = selected.drop(columns=['Weekly_Sales', 'Date'])
    
#     feature_names = joblib.load("model/feature_names.pkl")
#     features = features.reindex(columns=feature_names, fill_value=0)
    

#     prediction = model.predict(features)[0]
#     actual = selected['Weekly_Sales'].values[0]

#     # st.metric("📊 Predicted Weekly Sales", f"${prediction:,.2f}")
#     # st.metric("📌 Actual Sales", f"${actual:,.2f}")
#     col1, col2 = st.columns(2)
#     col1.metric("📊 Predicted Weekly Sales", f"${prediction:,.2f}")
#     col2.metric("📌 Actual Sales", f"${actual:,.2f}")


#     # Sales Alert
#     # if prediction > 50000:
#     #     st.error("⚠️ High Sales Alert! Monitor stock and logistics.")
#     # elif prediction < 1000:
#     #     st.warning("📉 Low Sales Alert! Investigate potential demand issues.")
#     if prediction > 50000:
#         st.error("🚨 **High Sales Alert!** Restock inventory and monitor logistics.")
#     elif prediction < 1000:
#         st.warning("📉 **Low Sales Warning!** Investigate promotions or local demand.")
#     else:
#         st.success("✅ Sales within normal range.")



# ✅ App Title with Translation
st.title(translations[language]["title"])
st.markdown(translations[language]["subtitle"])

# ✅ Sidebar Filters (Translated)
st.sidebar.header(translations[language]["filter_header"])
store = st.sidebar.selectbox(translations[language]["select_store"], sorted(final_df['Store'].unique()))
dept = st.sidebar.selectbox(translations[language]["select_dept"], sorted(final_df['Dept'].unique()))
date = st.sidebar.date_input(translations[language]["select_date"], final_df['Date'].min())

# ✅ Prediction Logic
selected = final_df[(final_df['Store'] == store) & (final_df['Dept'] == dept) & (final_df['Date'] == pd.to_datetime(date))]

if selected.empty:
    st.warning("No data found for this combination.")
else:
    features = selected.drop(columns=['Weekly_Sales', 'Date'])

    feature_names = joblib.load("model/feature_names.pkl")
    features = features.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(features)[0]
    actual = selected['Weekly_Sales'].values[0]

    col1, col2 = st.columns(2)
    col1.metric(translations[language]["predicted"], f"${prediction:,.2f}")
    col2.metric(translations[language]["actual"], f"${actual:,.2f}")

    # if prediction > 50000:
    #     st.error(translations[language]["high_alert"])
    # elif prediction < 1000:
    #     st.warning(translations[language]["low_alert"])
    # else:
    #     st.success("✅ Sales within normal range.")
    user_email = st.sidebar.text_input("📩 Enter your email for alerts (optional)")

    if prediction > 50000:
        st.error(translations[language]["high_alert"])
        if user_email:
            send_email_alert(
                subject="🚨 High Sales Alert",
                body=f"The predicted weekly sales for Store {store}, Department {dept} on {date} is ${prediction:,.2f}. Restock inventory accordingly.",
                to_email=user_email
            )
    elif prediction < 1000:
        st.warning(translations[language]["low_alert"])
        if user_email:
            send_email_alert(
                subject="📉 Low Sales Alert",
                body=f"The predicted weekly sales for Store {store}, Department {dept} on {date} is ${prediction:,.2f}. Investigate promotions or demand issues.",
                to_email=user_email
            )
    else:
        st.success("✅ Sales within normal range.")



# ========================
# 📊 ANALYTICS SECTION
# ========================

# 1️⃣ Weekly Sales Over Time
with st.expander("📅 Weekly Sales Over Time"):
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=final_df['Date'].min().date(),
        max_value=final_df['Date'].max().date(),
        value=(final_df['Date'].min().date(), final_df['Date'].max().date())
    )
    filtered = final_df[(final_df['Date'] >= pd.to_datetime(start_date)) & (final_df['Date'] <= pd.to_datetime(end_date))]
    st.line_chart(filtered.groupby("Date")["Weekly_Sales"].mean())

# 2️⃣ Store-wise Sales
with st.expander("🏪 Average Sales by Store"):
    year = st.selectbox("Select Year", sorted(final_df['Year'].unique()))
    store_sales = final_df[final_df['Year'] == year].groupby("Store")["Weekly_Sales"].mean()
    st.bar_chart(store_sales)

# 3️⃣ Department Sales
with st.expander("📊 Department Sales Comparison"):
    departments = st.multiselect("Select Departments", final_df['Dept'].unique(), default=[1, 2, 3])
    dep_data = final_df[final_df['Dept'].isin(departments)]
    dep_sales = dep_data.groupby("Dept")["Weekly_Sales"].mean()
    st.bar_chart(dep_sales)

# 4️⃣ Actual vs Predicted
with st.expander("🎯 Actual vs Predicted Sales"):
    try:
        y_test = pd.read_csv("y_test.csv")
        y_pred = pd.read_csv("y_pred.csv")

        n = st.slider("Select number of predictions to view", 10, 100, 50)
        chart_df = pd.DataFrame({
            "Actual": y_test.values[:n].flatten(),
            "Predicted": y_pred.values[:n].flatten()
        })
        st.line_chart(chart_df)
    except:
        st.info("Prediction data not available. Please ensure `y_test.csv` and `y_pred.csv` exist.")

# 5️⃣ Holiday vs Non-Holiday
with st.expander("🎉 Sales on Holidays vs Non-Holidays"):
    # Use either IsHoliday_x or IsHoliday_y
    holiday_col = 'IsHoliday_x' if 'IsHoliday_x' in final_df.columns else (
                  'IsHoliday_y' if 'IsHoliday_y' in final_df.columns else None)

    if holiday_col:
        include_holidays = st.checkbox("Include Holidays Only")
        filtered_holiday = final_df[final_df[holiday_col] == int(include_holidays)]

        # 📊 Boxplot: Weekly Sales on Holidays vs Non-Holidays
        st.subheader("📦 Sales Distribution (Box Plot)")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_holiday, x=holiday_col, y="Weekly_Sales", ax=ax)
        ax.set_title("Holiday vs Non-Holiday Weekly Sales")
        st.pyplot(fig)

        # 🧁 Interactive Pie Chart with Plotly
        st.subheader("🧁 Holiday Distribution (Interactive)")
        holiday_counts = final_df[holiday_col].value_counts().rename({0: "Non-Holiday", 1: "Holiday"}).reset_index()
        holiday_counts.columns = ['Holiday', 'Count']

        fig = px.pie(
            holiday_counts,
            names='Holiday',
            values='Count',
            color='Holiday',
            color_discrete_map={"Holiday": "#FF5722", "Non-Holiday": "#4CAF50"},
            title="Holiday vs Non-Holiday Week Distribution"
        )

        fig.update_traces(textinfo='percent+label', pull=[0.1, 0], hoverinfo='label+percent+value')
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Holiday information is missing in your dataset.")



# 6️⃣ Feature Importance (Interactive with Plotly)
with st.expander("🧠 Feature Importance (Interactive)"):
    try:
        import plotly.express as px

        # Extract feature importances
        importances = model.feature_importances_
        feature_names = model.feature_names_in_

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # User selects how many top features to display
        top_n = st.selectbox("Select number of top features to display", [5, 10, 15, 20], index=1)

        top_features = fi_df.head(top_n)

        # Create interactive plotly bar chart
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Bluered_r',
            title=f"Top {top_n} Most Important Features for Sales Prediction"
        )

        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=400 + top_n * 10,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Full Feature Ranking"):
            st.dataframe(fi_df.reset_index(drop=True))

    except Exception as e:
        st.warning("Unable to display feature importance.")
        st.text(str(e))

##email-alert
def send_email_alert(subject, body, to_email):
    sender_email = "your_email@example.com"
    sender_password = "your_app_password"  # Use app-specific password (e.g., from Gmail)

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print("📧 Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
