import requests
from datetime import datetime

import pandas
from prophet import Prophet

import streamlit
from plotly import graph_objects

def getTokenInfo(coin) -> dict:

    BASE_URL = "https://api.coingecko.com/api/v3"

    params = {
        "vs_currency": "USD",
        "ids": coin.get("id"),
    }

    response = requests.get(BASE_URL + f"/coins/markets", params=params).json()

    return {
        "price": response[0]["current_price"],
        "total_volume": response[0]["total_volume"],
        "total_supply": response[0]["total_supply"],
    }

def checkToken(symbol) -> bool:

    BASE_URL = "https://api.coingecko.com/api/v3/"

    response = requests.get(BASE_URL + f"/coins/list").json()

    coin_found = False

    for coin in response:
        if coin["symbol"].lower() == symbol.lower():
            coin_found = True
            break

    if not coin_found:  return None

    return coin

def extractData(coin:dict, interval:int) -> pandas.DataFrame:

    BASE_URL = "https://api.coingecko.com/api/v3"

    params = {
        "vs_currency": "USD",
        "days": interval
    }

    headers = {
        "accept": "application/json"
    }

    trainingData = requests.get(BASE_URL + f"/coins/{coin.get('id')}/market_chart",
        params=params, headers=headers).json()["prices"]

    trainingData = pandas.DataFrame(trainingData,
        columns=["timestamp", "price"]
    )

    trainingData = trainingData.rename(columns={"timestamp": "ds", "price": "y"})
    
    trainingData['ds'] = trainingData['ds'].apply(lambda x: datetime.fromtimestamp(
        int(x) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    
    return trainingData[['ds', 'y']]

def predictData(trainingData: pandas.DataFrame, periods:int, frequency: str) -> pandas.DataFrame:

    model = Prophet()
    model.fit(trainingData)

    future = model.make_future_dataframe(
        periods=periods, freq=frequency, include_history=True
    )
    forecastData = model.predict(future)

    return forecastData[['ds', 'yhat']]

freqPeriod = {
    "5 - Minutes": (1, '5T') ,  "15 - Minutes": (1, '15T'), "30 - Minutes" : (1, '30T'),
    "1 - Hour"   : (1, '1H') ,  "6 - Hours"   : (7, '6H') , "12 - Hours"   : (7, '12H'), 
    "1 - Day"    : (14, '1D'),  "7 - Days"    : (30, '7D'), "1 - Month"    : (365, '30D'),
}

streamlit.markdown("<h1>INSIGHT | AI Predictive Model 💫</h1>", unsafe_allow_html=True)
streamlit.markdown("<h3>SEE TRENDS BEFORE THEY HAPPEN 🪄</h3>", unsafe_allow_html=True)

with streamlit.form('token', clear_on_submit=True) as FORM:

    SYMBOL = streamlit.text_input("Enter token symbol")

    TIMEFRAME = streamlit.selectbox(
        "A glimpse into the future! 💫",
        tuple(freqPeriod.keys())
    )

    submit = streamlit.form_submit_button("INSIGHT 🔮")

if submit:

    coin = checkToken(SYMBOL)

    if coin:

        streamlit.success('Token found!', icon="✅")

        data = getTokenInfo(coin)

        with streamlit.spinner('Fetching data...'):
            trainingData = extractData(coin, freqPeriod[TIMEFRAME][0])

        streamlit.success('Data extracted successfully!', icon="✅")

        with streamlit.spinner('Predicting data...'):

            forecastData = predictData(trainingData, 1, freqPeriod[TIMEFRAME][-1])

        streamlit.success('Prediction complete!', icon="✅")

        info = getTokenInfo(coin)

        predictedPrice = forecastData['yhat'].iloc[-1]

        if predictedPrice <= 0: predictedPrice = 0

        predictedPricePercentage = round(abs(predictedPrice - info['price']) / info['price'] * 100, 2)

        if predictedPrice < info['price']:
            predictedPricePercentage = -1 * predictedPricePercentage

        if predictedPricePercentage > 0:
            text = f'''The current market price of {coin.get('name').upper()} appears bullish, with a projected surge of <code>{predictedPricePercentage}%</code> in the next {TIMEFRAME}.\
                \nThis hints at increased demand and a favorable market trend. Investors seeking high-risk investments may find the project enticing due to the short-term profit potential.\
                \nHowever, investors should always be wary of the associated risks when considering such a move.'''

        else:
            text = f'''The present market price of {coin.get('name').upper()} seems to be bearish, with a projected dip of <code>{predictedPricePercentage}%</code> in the next {TIMEFRAME}.\
                \nThis suggests a decrease in demand and negative market trend. Investors who are willing to take on high-risk investments may find the project unappealing due to its potential for short-term losses.\
                \nTherefore, it is advisable that investors exercise caution and carefully assess the potential risks involved in such a decision.\
                \nHowever, caution is recommended, and investors should be wary of the associated risks when considering such a move.'''

        from plotly import io

        fig = graph_objects.Figure()

        fig.update_layout(
            template=io.templates['plotly_dark'],
        )

        fig.add_trace(
            graph_objects.Scatter(
                x=forecastData['ds'],
                y=forecastData['yhat'],
                name='Predicted Price',
            )
        )

        fig.add_trace(
            graph_objects.Scatter(
                x=trainingData['ds'],
                y=trainingData['y'],
                name='Actual Price',
            )
        )

        streamlit.plotly_chart(fig)

        streamlit.markdown(f"<b>💠 Token ➤ <code>{coin.get('name')}</code>\n\n💵 Price ➤ <code>${info['price']}</code>\n\
                \n📊 Volume ➤ <code>${info['total_volume']}</code>\n\
                \n💰 Marketcap ➤ <code>${round(info['total_supply'] * info['price'])}</code>\n\n<i>{text}</i>\n\
                \n⏰ Prediction Time ➤ <code>{forecastData['ds'].iloc[-1]}</code>\n\
                \n🤑 Predicted Price ➤ <code>${predictedPrice} ({predictedPricePercentage}%)</code></b>", unsafe_allow_html=True)
    
    else:   streamlit.error('Token not found! 😕', icon="❌")
