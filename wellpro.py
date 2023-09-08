import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import datetime
from matplotlib.dates import DateFormatter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from openpyxl import load_workbook

alt.renderers.enable('default')
st.title('_PRODUCTION DATA ANALYSIS_ :blue[COOL] :sunglasses:')


def load_data():
    global df
    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file, encoding='latin-1')
        except Exception as e:
            print(e)
            df = pd.read_excel(upload_file, well_selected).fillna(0)
            df['Date'] = pd.to_datetime(df['Date'])
            # df.index = pd.to_datetime(df.index)
            # df.dropna(inplace=True)
            df.drop_duplicates(subset="Date", keep='last')
            df.sort_values("Date", inplace=True)
        return df


def hyperbolic(t, qi, di, b):
    """
  Hyperbolic decline function
  """
    import numpy as np
    return qi / (np.abs((1 + b * di * t)) ** (1 / b))


# function for hyperbolic cumulative production
def cumpro(q_forecast, qi, di, b):
    return (((qi ** b) / ((1 - b) * di)) * ((qi ** (1 - b)) - (q_forecast ** (1 - b))))


def q_hyp(t, qi, b, d):
    qfit = qi / (np.abs((1 + b * d * t)) ** (1 / b))

    return qfit


def hyp_fitter(q, t):
    # First we have to Normalize so that it converges well and quick.
    q_n = q / max(q)
    t_n = t / max(t)

    # curve-fit (optimization of parameters)
    params = curve_fit(q_hyp, t_n, q_n)
    [qi, b, d] = params[0]

    # These are for normalized t and q.
    # We must re-adjust for q and t (non-normalized)
    d_f = d / max(t)
    qi_f = qi * max(q)

    # Now we can use these parameters.
    q_hyp_fit = q_hyp(t, qi_f, b, d_f)

    return q_hyp_fit, params


tabs = ["Plot data", "DCA", "RTA", "About"]
st.sidebar.subheader("App Navigation")
page = st.sidebar.radio("Select your page", tabs)
upload_file = st.sidebar.file_uploader(label="Please upload your CSV or Excel file!", type=['csv', 'xlsx'])

if page == "Plot data":
    tab1, tab2 = st.tabs(["ACTUAL PERFORMANCE", "PRODUCTION SUMMARY"])
    with tab1:
        try:
            # Loading the  dataset
            file = pd.ExcelFile(upload_file)
            well_name = file.sheet_names
            st.header('Select well', divider='rainbow')
            well_selected = st.selectbox("", well_name)
            df = load_data()
            brush = alt.selection_interval()
            df['WCT'] = df['Rate Water'] * 100 / df['Rate Liquid']
            df['GOR'] = df['Rate Gas'] * 1000 / df['Rate Oil']
            st.write(df)
            domain_t = ["2015-01-01", "2022-12-31"]
            domain_o = [0, 4000]
            domain_w = [0, 5000]
            domain_g = [0, 20]
            selection = alt.selection_multi(fields=['Rate Oil'], on='click')
            highlight = alt.selection(type='single', on='mouseover', fields=['Rate Oil', 'Rate Gas'], nearest=True)
            base = alt.Chart(df).encode(
                alt.X('Date', axis=alt.Axis(format="%B %Y", labelFontSize=18, grid=False))).properties(
                width=800,
                height=400,
            )
            a = base.mark_line(opacity=1, color='Green').encode(
                alt.Y('Rate Oil', title="Oil Rate [stb/d]", axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14),
                      scale=alt.Scale(domain=domain_o))
            ).interactive()
            b = base.mark_line(opacity=1, color='blue').encode(
                alt.Y('Rate Water', title="Water Rate [stb/d]",
                      axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14), scale=alt.Scale(domain=domain_o))
            ).interactive()
            c = base.mark_line(opacity=1, color='red').encode(
                alt.Y('Rate Gas', title="Gas Rate [MMscf/d]",
                      axis=alt.Axis(offset=00, titleFontSize=18, labelFontSize=14), scale=alt.Scale(domain=domain_g))
            ).interactive()

            d1 = alt.layer(b, a).resolve_scale(y='shared')
            d2 = alt.layer(c, d1).resolve_scale(y='independent')

            st.header('Production Performance ' + well_selected, divider='rainbow')

            st.altair_chart(d2, theme=None, use_container_width=True)
        except Exception as e:
            print(e)
            st.write("Please upload file to the application")
    with tab2:
        # Use the native Altair theme.
        # st.altair_chart(chart, theme=None, use_container_width=True)
        st.write("Please upload file to the application")

if page == "DCA":
    file = pd.ExcelFile(upload_file)
    well_name = file.sheet_names
    st.header('Select well', divider='rainbow')
    well_selected = st.selectbox("", well_name)
    df = load_data()
    #non =df['Rate Oil'].idxmax()
    non = df.loc[df['Rate Oil'].ne(0), 'Rate Oil'].first_valid_index()

    df1 = df.drop(df.index[0:non])
    st.write(df1)
    t = df1['Date']
    q = df1['Rate Oil']

    fig1 = plt.figure(figsize=(10, 7))
    plt.step(t, q, color='Green')
    plt.title('Production Rate from', size=20, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Rate Oil (stb/d)')
    plt.xlim(min(t), max(t))
    plt.ylim(ymin=0)
    st.header('Oil rate plot', divider='orange')
    st.pyplot(fig1)

    # subtract one datetime to another datetime
    timedelta = [j - i for i, j in zip(t[:-1], t[1:])]
    timedelta = np.array(timedelta)
    timedelta = timedelta / datetime.timedelta(days=1)
    t_qmax = q.idxmax()
    st.write(t_qmax)
    st.write(max(q))

    # take cumulative sum over timedeltas
    t = np.cumsum(timedelta)
    t = np.append(t_qmax, t)
    t = t.astype(float)

    # normalize the time and rate data
    t_normalized = t / max(t)
    q_normalized = q / max(q)

    # fitting the data with the hyperbolic function
    popt, pcov = curve_fit(hyperbolic, t_normalized, q_normalized)

    qi, di, b = popt

    # de-normalize qi and di
    qi = qi * max(q)
    di = di / max(t)

    # forecast gas rate until 1,500 days
    t_forecast = np.arange(4000)
    q_forecast = hyperbolic(t_forecast, qi, di, b)

    # forecast cumulative production until 1,500 days
    Qp_forecast = cumpro(q_forecast, qi, di, b)

    # plot the production data with the forecasts (rate and cum. production)
    fig2 = plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, q, '.', color='green', label='Production Data')
    plt.plot(t_forecast, q_forecast, label='Forecast')
    plt.title('Oil Production Rate Result of DCA', size=13, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Rate (stb/d)')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_forecast, Qp_forecast)
    plt.title('Oil Cumulative Production Result of DCA', size=13, pad=15)
    plt.xlabel('Days')
    plt.ylabel('Production (STB)')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    st.header('Prediction Oil rate plot', divider='orange')
    st.pyplot(fig2)

    st.write('Initial production rate:', np.round(qi, 3), 'stb/d')
    st.write('Initial decline rate:', np.round(di, 3), 'stb/d')
    st.write('Decline coefficient:', np.round(b, 3))

    # st.write("Please upload file to the application")

if page == "RTA":
    st.write("Please upload file to the application")

if page == "About":
    st.write("This app is built by vulm. Feel free to contact me via email: lmvu103@gmail.com")
