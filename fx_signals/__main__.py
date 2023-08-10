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

def prepare_watchlist(api: v20.Context, args) -> dict:    
    available_intruments = api.account.instruments(
        args.config.active_account,
    )
    
    pairs = dict()
    for instr in available_intruments.get('instruments'):
        if instr.type != 'CURRENCY':
            continue
        pairs[instr.name] = instr
    
    us_pairs = {
        instr_name: instr for instr_name, instr in pairs.items() if 'USD' in instr_name
    }
    
    return us_pairs

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
    granularities: list = kwargs.get('granularities', ['M15', 'H1', 'H4', 'D'])
    dataframes = dict()
    return dataframes

def main():
    parser = argparse.ArgumentParser()

    config.add_argument(parser)

    args = parser.parse_args()
    api: v20.Context = args.config.create_context()
    streaming_api = args.config.create_streaming_context()
    
    pairs: dict = prepare_watchlist(api, args)
    wb: xw.Book = get_workbook()
    while True:
        start = time.time()
        cycle(pairs, api, args, wb)
        end = time.time()
        time.sleep(1 - (end - start))
        # print('Cycle took: ', end - start)

if __name__ == "__main__":
    main()