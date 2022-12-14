import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


f = pd.ExcelFile('Daily_production.xlsx')


# df.dropna(inplace=True)
# df['Date'] = pd.to_datetime(df['Date'])

brush = alt.selection_interval()

# base = alt.Chart(df).encode(alt.X('Date', axis=alt.Axis(format="%B %Y", labelFontSize=18, grid=False))).properties(
#     width=800,
#     height=400,
# )

# a = base.mark_point(opacity=1, color='green').encode(
#     alt.Y('Oil', axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14)), )
# b = base.mark_line(opacity=1, color='blue').encode(
#     alt.Y('Water', axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14)), )
# c = base.mark_point(opacity=1, color='red').encode(
#     alt.Y('Gas', axis=alt.Axis(offset=60, titleFontSize=18, labelFontSize=14)), )
well_names = f.sheet_names
well_selected = st.selectbox("Select well:", well_names)
df = f.parse(sheet_name=well_selected)
st.dataframe(df)
chart = alt.Chart(df).mark_point(filled=True, size=100).encode(
    x=alt.X('Date'),
    y=alt.Y('Oil', ),
    color=alt.condition(brush, 'group:N', alt.value('lightgray'))
).add_selection(
    brush
)
t = df['Date']
q = df['Oil']


## Decline Curves:
def Exponential(t, di):
    qi = Data['Production'].max()
    return qi * np.exp(-di * t)


def Hyperbolic(t, b, di):
    qi = Data['Production'].max()
    return qi / ((1.0 + b * di * t) ** (1 / b))


def Harmonic(t, qi, di):
    return qi / (1 + di * t)


# simple graph function
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Production (bbls)")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

# d = alt.layer(a, b, c).resolve_scale(y='independent')
d = chart
st.altair_chart(d, use_container_width=True)
