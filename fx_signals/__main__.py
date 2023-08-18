import argparse
from fx_signals.common import config
from fx_signals.api import get_df
from fx_signals.utils.excel_helper import cycle

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

def prepare_watchlist(api: v20.Context, args, currencies: list) -> dict:    
    available_intruments = api.account.instruments(
        args.config.active_account,
    )
    
    pairs = dict()
    for instr in available_intruments.get('instruments'):
        if instr.type != 'CURRENCY':
            continue
        base, quote = instr.name.split('_')
        if base in currencies and quote in currencies:
            pairs[instr.name] = instr
    
    return pairs

def get_workbook() -> xw.Book:
    helper_workbook = os.path.expanduser('~/fx_helper.xlsx')
    if not os.path.exists(helper_workbook):
        wb = xw.Book()
        wb.save(helper_workbook)
    wb = xw.Book(helper_workbook)
    return wb

def df_key(pair, granularity):
    return f'{pair}_{granularity}'

def warmup(pairs, api, *args, **kwargs) -> dict:
    granularities: list = kwargs.get('granularities', ['M1', 'M15', 'H1', 'H4', 'D'])
    dataframes = dict()
    incomplete_candles = list()
    for pair in pairs:
        for granularity in granularities:
            key = df_key(pair, granularity)
            dataframes[key], incomplete = get_df(api, pair, granularity, count=200)
            incomplete_candles.extend(incomplete)
    return dataframes, incomplete_candles

def main():
    parser = argparse.ArgumentParser()

    config.add_argument(parser)

    args = parser.parse_args()
    api: v20.Context = args.config.create_context()
    streaming_api = args.config.create_streaming_context()
    
    currencies_to_watch = [
        'EUR', 'GBP',
        'AUD', 'NZD',
        'CAD', 'CHF',
        'JPY', 'USD',
        'SGD'
    ]
    
    watchlist = prepare_watchlist(api, args, currencies_to_watch)
    candles, incomplete = warmup(
        watchlist,
        api,
        granularities=['M1', 'M15', 'H1', 'H4', 'D']
    )
    print(incomplete)
    wb: xw.Book = get_workbook()
    while True:
        start = time.time()
        cycle(watchlist, api, args, wb)
        end = time.time()
        delta = end - start
        if delta < 1:
            time.sleep(1 - delta)
        print(f'Cycle took: {delta} seconds')

if __name__ == "__main__":
    main()