#!/usr/bin/env python

from dataclasses import astuple
import numpy as np
import pandas as pd
import py_vollib
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes.greeks.analytical import delta
from py_vollib.black_scholes.greeks.analytical import gamma
from py_vollib.black_scholes.greeks.analytical import theta
from py_vollib.black_scholes.greeks.analytical import vega
from scipy.optimize import fsolve
from scipy.stats import norm
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb

import betterproto

import asyncio


option_strikes = [90, 95, 100, 105, 110]

SELL = 1
BUY = 1
SELL_RANGE = .3
BUY_RANGE = .3

class Case2ExampleBot(UTCBot):
    """
    An example bot for Case 2 of the 2021 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    """

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """

        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}
        self.underlying_price_hist = []
        self.vol_hist = []
        self.option_hist = {}
        self.delta_lit, self.gamma_lit, self.vega_lit, self.theta_lit = 2000, 5000, 1000000, 500000

        self.positions["UC"] = 0
        for strike in option_strikes:
            for flag in ["C", "P"]:
                self.positions[f"UC{strike}{flag}"] = 0
                self.option_hist[f"UC{strike}{flag}"] = []

        # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0

        # Stores the current value of the underlying asset
        self.underlying_price = 100
        # underlying_price_hist.append(self.underlying_price)

    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """
        # if (len(self.underlying_price_hist) < 10):
        #     return 0.2
        stdv_arr = np.array(self.underlying_price_hist)
        self.vol_hist.append(np.std(stdv_arr))
        if (len(self.underlying_price_hist) > 5):
            return np.std(stdv_arr)
        return 0.2

    def compute_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters. Some
        important questions you may want to think about are:
            - What are the units associated with each of these quantities?
            - What formula should you use to compute the price of the option?
            - Are there tricks you can use to do this more quickly?
        You may want to look into the py_vollib library, which is installed by default in your
        virtual environment.
        """
        S, K, T, r, sigma = underlying_px, strike_px, time_to_expiry, 0, volatility
        def d1(S,K,T,r,sigma):
            return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
        def d2(S,K,T,r,sigma):
            return d1(S,K,T,r,sigma)-sigma*np.sqrt(T)
        def bs_call(S,K,T,r,sigma):
            return S*norm.cdf(d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
        def bs_put(S,K,T,r,sigma):
            return K*np.exp(-r*T)-S+bs_call(S,K,T,r,sigma)
        if (flag == "C"):
            return bs_call(S,K,T,r,sigma)
        if (flag == "P"):
            return bs_put(S,K,T,r,sigma)
        raise Exception("False flag")
    
    def greeks_comp(self, asset_name, flag, S, K, t, r, sigma):
        delta_c = self.positions[asset_name] * delta(flag.lower(), S, K, t, r, sigma) 
        theta_c = self.positions[asset_name] * theta(flag.lower(), S, K, t, r, sigma)
        gamma_c = self.positions[asset_name] * gamma(flag.lower(), S, K, t, r, sigma)
        vega_c = self.positions[asset_name] * vega(flag.lower(), S, K, t, r, sigma)
        # print('delta, theta, gamma, vega', delta(flag.lower(), S, K, t, r, sigma) + self.positions["UC"], theta(flag.lower(), S, K, t, r, sigma),\
        #     gamma(flag.lower(), S, K, t, r, sigma), vega(flag.lower(), S, K, t, r, sigma))
        return (delta_c, theta_c, gamma_c, vega_c)

    def iv_call(S,K,T,r,C):
        return fsolve((lambda sigma: np.abs(compute_options_price("C", S,K,T,sigma) - C)), [1])[0]
                      
    def iv_put(S,K,T,r,P):
        return fsolve((lambda sigma: np.abs(compute_options_price("P", S,K,T,sigma) - C)), [1])[0]


    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        global BUY, SELL, BUY_RANGE, SELL_RANGE

        # What should this value actually be?
        time_to_expiry = (26 - self.current_day) / 252
        vol = self.compute_vol_estimate()
        requests = []

        with open('case2_params') as params:
            for param in params.readlines():
                pairs = param.split(':')
                if len(pairs) == 2:
                    try:
                        if pairs[0].strip() == 'BUY_RANGE':
                            BUY_RANGE = float(pairs[1].strip())
                        elif pairs[0].strip() == 'SELL_RANGE':
                            SELL_RANGE = float(pairs[1].strip())
                        elif pairs[0].strip() == 'BUY':
                            BUY = int(pairs[1].strip())
                        elif pairs[0].strip() == 'SELL':
                            SELL = int(pairs[1].strip())
                    except ValueError:
                        pass

        # requests.append(
        #     self.place_order(
        #         "UC",
        #         pb.OrderSpecType.MARKET,
        #         pb.OrderSpecSide.BID,
        #         5
        #     )
        # )

        # requests.append(
        #     self.place_order(
        #         "UC",
        #         pb.OrderSpecType.MARKET,
        #         pb.OrderSpecSide.ASK,
        #         5
        #     )
        # )

        delta_c, theta_c, gamma_c, vega_c = 0, 0, 0, 0
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                a1, a2, a3, a4 = self.greeks_comp(asset_name, flag, self.underlying_price, strike, time_to_expiry, 0, vol)
                delta_c, theta_c, gamma_c, vega_c = a1 + delta_c, a2 + theta_c, a3 + gamma_c, a4 + vega_c
        delta_c = abs(delta_c * 100 + self.positions["UC"])
        theta_c = abs(theta_c * 100)
        gamma_c = gamma_c * 100
        vega_c = vega_c * 100

        print("delta, theta, gamma, vega", delta_c, theta_c, gamma_c, vega_c)
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                self.positions[asset_name]
                theo = self.compute_options_price(
                    flag, self.underlying_price, strike, time_to_expiry, vol
                )
                theo = round(theo, 1)
                
                self.option_hist[asset_name].append(theo)
                # print('mine', theo, 'lib', bs(flag.lower(), strike, self.underlying_price, time_to_expiry, 0, vol))
                # if there's high risk, what can we do
                # print(asset_name, delta(flag, self.underlying_price, strike, time_to_expiry, 0, vol))
                op_hist = np.array(self.option_hist[asset_name])
                # tmr check abs and choose buy or sell the stuff
                # if (abs(self.positions[asset_name]) > 250):
                #     sell, buy = 15, 1
                #     sell_range, buy_range = 4, -10
                # elif (abs(self.positions[asset_name]) > 200):
                #     sell, buy = 15, 1
                #     sell_range, buy_range = 2, -8
                # elif (abs(self.positions[asset_name]) > 150):
                #     sell, buy = 10, 1
                #     sell_range, buy_range = 1, -5
                # elif (theo < np.average(op_hist) and np.nanstd(op_hist) > 0.45):
                #     #it shows it goes into pandemic
                #     sell, buy = 1, 6
                #     sell_range, buy_range = 0.1, 0.5
                # elif (theo < np.average(op_hist)):
                #     sell, buy = 1, 4
                #     sell_range, buy_range = 0.2, 0.3
                # else:
                #     sell, buy = 1, 2
                #     sell_range, buy_range = 0.3, 0.3
                if (delta_c> self.delta_lit or theta_c> self.theta_lit or gamma_c> self.gamma_lit or vega_c> self.vega_lit):
                    SELL, BUY = 5, 1
                    SELL_RANGE, BUY_RANGE = 0.7, -1.5
                    
                    requests.append(
                    self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        SELL,
                        theo + SELL_RANGE,
                    )
                    )
                    continue
                print("sell:", SELL, 'buy', BUY, 'sell_range', SELL_RANGE, 'buy_range', BUY_RANGE)
                requests.append(
                    self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        BUY,  # How should this quantity be chosen?
                        theo - BUY_RANGE,  # How should this price be chosen?
                    )
                )
                requests.append(
                    self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        SELL,
                        theo + SELL_RANGE,
                    )
                    )
                # bid mean you're selling

        # optimization trick -- use asyncio.gather to send a group of requests at the same time
        # instead of sending them one-by-one
        responses = await asyncio.gather(*requests)
        for resp in responses:
            assert resp.ok

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)
            # if (len(self.underlying_price_hist) > 5):
            #     print((self.underlying_price_hist[len(self.underlying_price_hist) - 5:][0] - self.underlying_price_hist[len(self.underlying_price_hist) - 5:][4]/4))
            print(self.positions, self.current_day)

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "market_snapshot_msg":
            # When we receive a snapshot of what's going on in the market, update our information
            # about the underlying price.
            book = update.market_snapshot_msg.books["UC"]

            # Compute the mid price of the market and store it
            temp = self.underlying_price
            self.underlying_price = (float(book.bids[0].px) + float(book.asks[0].px)) / 2
            self.underlying_price_hist.append(round(self.underlying_price, 1))

            await self.update_options_quotes()

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case) 
            self.current_day = float(update.generic_msg.message)


if __name__ == "__main__":
    test = start_bot(Case2ExampleBot)