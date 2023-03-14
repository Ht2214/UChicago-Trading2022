# UChicago-Trading 2022
 https://tradingcompetition.uchicago.edu/

We represented UC Berkeley to compete in UChicago Trading Competition 2022 and won 3rd place in portfolio optimization. This competition involved 3 separate aspects of trading: market making, trading options, and portfolio management. There we developed a significant understanding of trading strategies by coding out our algorithms.

## Case Strategies
### Case 1 
Cae 1 is about market making on lumber futures. We received daily price data on lumber and monthly precipitation. With the given data, we attempted to predict future prices of lumber given the trend of precipitation across the entire year. We started with naive SVM, LSTM, and ARIMA. Later we moved to advance deep learning models, which didn't perform well due to the lack of data and over-fitting. After thorough experiment and evaluation, we adopted SARIMA, Seasonal Autoregressive Integrated Moving Average to predict the fair price of future by considering the seasonal trend of precipitation. 

To be more specific about market making strategy, we use the fair price we predicted by our model and determine the spread by backtesting. Besides that, we also incorporate manual modification for trading bot parameters: spread size, size to buy, and sell. Therefore, we can modify the parameter base on our position and prevent the model goes wrong given the dynamic of the market. 


### Case 2
Case 2 is about option pricing. We adopted a similar strategy with case 1 by predicting the general trend by SARIMA to determine the true price of options then trade according to option greek. With the predicted price, we apply the Black Scholes model to implement option trading. In addition, we also incorporate the same parameter modification feature, by letting our model read the editable parameter file.

### Case 3
Case 3 is about portfolio optimization. At first, we studied the fundamental of modern portfolio theory from Investopedia to research papers published by UPenn, Berkeley, and Duke. Finally, we decided to implement the Black Litterman model as it's an improved version of the Markowitz Model, which incorporates investors’ personal views concerning the allocation of assets in a portfolio. The implementation and mathematics behind are referring to A STEP-BY-STEP GUIDE TO THE BLACK-LITTERMAN MODEL[1].


[1]:https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf 
