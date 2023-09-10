import argparse
import asyncio

import pytz
from fx_signals.common import config
from fx_signals.api import get_df
from fx_signals.utils.gui import acycle
from async_v20 import OandaClient
from nicegui import ui

import os
import pandas as pd

import xlwings as xw
import datetime
import time

import v20
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_workbook() -> xw.Book:
    helper_workbook = os.path.expanduser('~/fx_helper.xlsx')
    if not os.path.exists(helper_workbook):
        wb = xw.Book()
        wb.save(helper_workbook)
    wb = xw.Book(helper_workbook)
    return wb

def df_key(pair, granularity):
    return f'{pair}_{granularity}'

async def aprepare_watchlist(client: OandaClient, currencies: list) -> dict:    
    # res = await client.account_instruments()
    
    excluded_markets = [
    ]
    pairs = dict()
    other_instrument_types = set()
    for instr in client.instruments:
        if instr.type != 'CURRENCY':
            continue
            include = False
            for tag in instr.tags:
                if tag['type'] != 'ASSET_CLASS':
                    continue
                if tag['name'] == 'INDEX':
                    include = True
                    break
            if not include:
                continue
        base, quote = instr.name.split('_')
        if base in currencies and quote in currencies:
            pairs[instr.name] = instr
        if instr.type != 'CURRENCY':
            pairs[instr.name] = instr
    return pairs

async def awarmup(client: OandaClient, pairs, **kwargs) -> dict:
    granularities: list = kwargs.get('granularities', ['H1', 'H4', 'D'])
    coroutines = list()
    for pair in pairs:
        for granularity in granularities:
            coroutines.append(client.get_candles(
                pair,
                'M',
                granularity,
                count=300
            ))
    responses = await asyncio.gather(*coroutines)
    candles_dict = dict()
    for response in responses:
        instrument = response.get('instrument')
        granularity = response.get('granularity')
        candles = response.get('candles')
        df = candles.dataframe()
        df = df.drop(df[df.complete == False].index)
        key = df_key(instrument, granularity)
        # print(candles_dict)
        candles_dict[key] = df
        # print(candles_dict)
    return candles_dict

async def helper_loop(client: OandaClient, state, **kwargs):
    wb: xw.Book = get_workbook()
    
    while True:
        start = time.time()
        await acycle(state['pairs'], client, wb)
        end = time.time()
        delta = end - start
        if delta < 1:
            time.sleep(1 - delta)
        print(f'Cycle took: {delta} seconds')

def get_ny_time():
    singapore = pytz.timezone('Asia/Singapore')
    new_york = pytz.timezone('America/New_York')
    sg_time = datetime.datetime.now(singapore)
    ny_time = sg_time.astimezone(new_york)
    return ny_time
        
def check_granularities():
    ny_time = get_ny_time()
    granularities_mapping = {
        'M1': True,
        'M5': ny_time.minute % 5 == 0,
        'M15': ny_time.minute % 15 == 0,
        'H1': ny_time.minute == 0,
        'H4': ny_time.minute == 0 and ny_time.hour % 4 == 0,
        'D': ny_time.minute == 0 and ny_time.hour == 0
    }
    return granularities_mapping
        
async def datafeed_loop(client: OandaClient, state, **kwargs):
    if not state['initialized']:
        logger.info('Not initialized')
        return
        
    for key in state['granularities']:
        granularities = check_granularities()
        if not granularities[key]:
            logger.debug(f'Not time for {key}')
        else:
            logger.debug(f'Getting {key}')
        await asyncio.sleep(1)
    

async def amain():
    init_args = {
        'rest_host': 'api-fxtrade.oanda.com',
        'rest_port': 443,
        'stream_host': 'stream-fxtrade.oanda.com',
        'stream_port': 443,
        'account_id': '001-003-10018443-003',
        'max_requests_per_second': 120
    }
    currencies_to_watch = [
        'EUR','USD','JPY', 'AUD', 'GBP', 'CAD', 'CHF', 'NZD'
    ]
    async with OandaClient(**init_args) as client:
        watchlist = await aprepare_watchlist(client, currencies_to_watch)
        # return
        state = dict()
        state['granularities'] = ['M1', 'M5', 'M15', 'H1', 'H4', 'D']
        state['pairs'] = watchlist
        state['initialized'] = False
        candles_dict = await awarmup(
            client,
            state['pairs'],
            granularities=state['granularities']
        )
        state['initialized'] = True
        state['candles'] = candles_dict
        
        candles_dict = state['candles']
        logger.info(candles_dict.keys())
        await asyncio.gather(
            datafeed_loop(client, state),
            helper_loop(client, state)
        )
        while True:
            await asyncio.sleep(1)
