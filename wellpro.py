import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

alt.renderers.enable('altair_viewer')

# read the file
#file = r'well_hist.xlsx'
#df = pd.read_excel(file, index_col=False)

# Loading the  dataset
df = pd.read_excel('DAILY_PRODUCTION_PLOT.xlsx', 'TL_1P').fillna("")
test = df.astype(str)
st.dataframe(test)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

interval = alt.selection_interval()
domain_t = ["2019-01-01", "2019-12-31"]
domain_o = [1500, 0]

base = alt.Chart(df).encode(alt.X('Date', axis=alt.Axis(format="%B %Y", labelFontSize=18, grid=False))).properties(
    width=800,
    height=400,
)

a = base.mark_line(opacity=0.6, color='Green').encode(
    alt.Y('Oil', axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14),scale=alt.Scale(domain=domain_o))
).interactive()

b = base.mark_line(opacity=0.6, color='blue').encode(
    alt.Y('Water', axis=alt.Axis(offset=0, titleFontSize=18, labelFontSize=14)),
)
c = base.mark_line(opacity=0.6, color='red').encode(
    alt.Y('Gas', axis=alt.Axis(offset=60, titleFontSize=18, labelFontSize=14)),
)

d = alt.layer(a, b, c).resolve_scale(y='independent')

st.altair_chart(d, use_container_width=True)

