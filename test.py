import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure
# from bokeh.models import CrosshairTool, Span, Button
from bokeh.models import WheelZoomTool, PanTool, CrosshairTool, ResetTool, HoverTool, Span, NumeralTickFormatter
import yfinance as yf
# from bokeh.io import output_notebook
import pandas_ta as ta

pn.extension()

def cmfx(high, low, close, volume, open_=None, length=None, offset=None, **kwargs):
    """Indicator: Chaikin Money Flow (CMF)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 20
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    high = ta.verify_series(high, _length)
    low = ta.verify_series(low, _length)
    close = ta.verify_series(close, _length)
    volume = ta.verify_series(volume, _length)
    offset = ta.get_offset(offset)

    if high is None or low is None or close is None or volume is None: return

    # Calculate Result
    if open_ is not None:
        open_ = ta.verify_series(open_)
        ad = ta.non_zero_range(close, open_)  # AD with Open
    else:
        ad = 2 * close - (high + low)  # AD with High, Low, Close

    ad *= volume / ta.non_zero_range(high, low)

    def test(s):
        x = np.array(range(5, length+5))
        # x = np.log(x)
        # print(x)
        # print(s)
        x = x/x.sum()
        # print(x * s)
        return (s*x).sum()

    cmf = ad.rolling(length, min_periods=min_periods).apply(test)#.sum()
    cmf /= volume.rolling(length, min_periods=min_periods).apply(test)#.sum()
    # cmf = ad.rolling(length, min_periods=min_periods).sum()
    # cmf /= volume.rolling(length, min_periods=min_periods).sum()


    # Offset
    if offset != 0:
        cmf = cmf.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cmf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cmf.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    # cmf.name = f"CMF_{length}"
    # cmf.category = "volume"

    return cmf

# Function to create a sine wave plot
def create_plot(symbol, sample_len, sma_len):
    df = yf.download(f'{symbol}', period='5y')
    df.reset_index(inplace=True)
    
    col = pn.Column()
    # hair = CrosshairTool(overlay=[600, 400])
    width = Span(dimension="width")
    height = Span(dimension="height")

    main = figure(title=f'{symbol}', width=600, height=400, x_axis_type="datetime", x_range=(df.Date.iloc[-500], df.Date.iloc[-1]))
    main.line(x='Date', y='Adj Close', source=df)
    main.add_tools(CrosshairTool(overlay=[width, height]))
    col.append(main)

    f = figure(x_axis_type="datetime", height=150, x_range=main.x_range)
    f.add_layout(Span(location=0, dimension='width'))

    # ticker = yf.Ticker(f'{symbol}')
    # s = pd.DataFrame(ticker.dividends)
    # s.reset_index(inplace=True)
    # f.line(x=s['Date'], y=s['Dividends'], color='blue')
    # f.add_tools(CrosshairTool(overlay=[width, height]))
    # col.append(f)

    # volume = figure(x_axis_type="datetime", height=150, x_range=main.x_range)
    # volume.vbar(x=df['Date'], width=0.5, bottom=0, top=df['Volume'])
    # volume.add_tools(CrosshairTool(overlay=[width, height]))
    # col.append(volume)
    
    f = figure(x_axis_type="datetime", height=150, x_range=main.x_range)
    f.add_layout(Span(location=0, dimension='width'))
    f.yaxis[0].formatter = NumeralTickFormatter(format="0.00000")
    # s = ta.cmf(open=df.Open, high=df.High, low=df.Low, close=df.Close, volume=df.Volume, length=sample_len)
    s = cmfx(open=df.Open, high=df.High, low=df.Low, close=df.Close, volume=df.Volume, length=sample_len)
    s = ta.sma(s, length=sma_len)
    f.line(x=df['Date'], y=s, color='red')
    f.add_tools(CrosshairTool(overlay=[width, height]))
    col.append(f)

    f = figure(x_axis_type="datetime", height=150, x_range=main.x_range)
    f.add_layout(Span(location=0, dimension='width'))
    s = s.diff()
    f.line(x=df['Date'], y=s, color='red')
    f.add_tools(CrosshairTool(overlay=[width, height]))
    col.append(f)

    # f = figure(x_axis_type="datetime", height=150, x_range=main.x_range)
    # f.add_layout(Span(location=0, dimension='width'))
    # s = s.diff()
    # f.line(x=df['Date'], y=s, color='red')
    # f.add_tools(CrosshairTool(overlay=[width, height]))
    # col.append(f)

    # print(df.Close.tail(20))

    # record=[]
    # index = 0
    # dir = s.iloc[len(s)-1] - s.iloc[len(s)-2]
    # for i in range(1, 100):
    #     record.append(s[len(s)-i])
    #     if (s[len(s)-i] - s[len(s)-(i+1)]) * dir < 0:
    #         index = i
    #         break
    
    # main.line(x=[df.Date[len(s)-index], df.Date[len(s)-index]], y=[-np.inf,np.inf])
    # main.add_layout(Span(location=df.Date[len(s)-index], dimension='height'))
    main.add_layout(Span(location=df.Date[len(s)-sample_len], dimension='height'))
    # col.append(f'Record: {record}')
    # col.append(f'Index: {index}')

    # point = s[len(s)-index]
    # if dir > 0:
    #     col.append(f'Lowest CMF: {point}')
    # else:
    #     col.append(f'Highest CMF: {point}')

    # col.append(f'The price = {df.Close.iloc[len(s)-index]}')
    
    return pn.Column(col)

# Panel slider widget
symbol_input = pn.widgets.TextInput(name='Symbol', placeholder='Enter symbol', value='2330.tw')
sample_len_slider = pn.widgets.IntSlider(name='Sample Length', start=2, end=200, step=1, value=50)
sma_len_slider = pn.widgets.IntSlider(name='SMA Length', start=2, end=20, step=1, value=10)

# Panel interactive function
@pn.depends(symbol_input, sample_len_slider.param.value, sma_len_slider.param.value)
def update_plot(symbol_input, sample_len_slider, sma_len_slider):
    plot = create_plot(symbol_input, sample_len_slider, sma_len_slider)
    return plot

# Create a Panel layout
layout = pn.Row(
    pn.Column(
        symbol_input,
        sample_len_slider,
        sma_len_slider,
    ),
    pn.Column(
        pn.panel(update_plot)
    )
)

layout.servable()
