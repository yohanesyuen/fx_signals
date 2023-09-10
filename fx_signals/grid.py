import argparse
import asyncio
import jsonpickle
import logging
import math

import pandas as pd
import talib as ta

from collections import OrderedDict
from decimal import Decimal

from async_v20 import OandaClient

from fx_signals.api import get_df
from fx_signals.common import config
from fx_signals.utils.excel_helper import acycle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Oanda client
def get_client():
    init_args = {
        'rest_host': 'api-fxtrade.oanda.com',
        'rest_port': 443,
        'stream_host': 'stream-fxtrade.oanda.com',
        'stream_port': 443,
        'account_id': '001-003-10018443-003',
        'max_requests_per_second': 120
    }
    return OandaClient(**init_args)


def df_key(pair, granularity):
    return f'{pair}_{granularity}'

async def aprepare_watchlist(client: OandaClient, currencies: list) -> dict:        
    pairs = dict()
    for instr in client.instruments:
        if instr.type != 'CURRENCY':
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

# Convert number to Decimal with precision based on pip_location
def to_decimal(number, pip_location):
    precision = abs(pip_location)
    format_str = f"0.{ '0' * precision }1"
    return Decimal(str(number)).quantize(Decimal(format_str))

# Convert price to pips
def price_to_pips(price, pip_location):
    ret = to_decimal(price, pip_location) / Decimal(10 ** pip_location)
    return to_decimal(ret, pip_location)

# Convert pips to price
def pips_to_price(pips, pip_location):
    price = Decimal(str(pips * 10 ** pip_location))
    return price.quantize(Decimal('0.' + '0' * abs(pip_location) + '1'))

# Get mid price
def get_mid(price, pip_location=-4):
    bid = price.bids[0].price
    ask = price.asks[0].price
    return to_decimal((bid + ask) / 2, pip_location)

# Get instrument information
def get_instrument_infos(client):
    return {instr.name: instr for instr in client.instruments}

# Load instrument grid
def load_instrument_grid(instrument):
    pass

def save_instrument_grid(instrument, grids):
    jsonpickle.dumps(grids)

# Generate grid lines
def generate_grids(mid_in_pips, grid_size, pip_location=-4):
    base_grid = math.floor(mid_in_pips / grid_size) * grid_size - grid_size * 10
    grids = [base_grid + i * grid_size for i in range(21)]
    grids = [pips_to_price(grid, pip_location) for grid in grids]
    grids = OrderedDict.fromkeys(grids)
    for k in grids.keys():
        grids[k] = {
            'long': None,
            'short': None
        }
    return grids

# Asynchronous function to generate grid
async def generate_grid(client: OandaClient, instrument, grid_size=10):
    instrument_infos = get_instrument_infos(client)
    pip_location = instrument_infos[instrument].pip_location
    pricing = (await client.get_pricing(instrument)).get('prices')
    mid = get_mid(pricing[0])
    mid_in_pips = price_to_pips(mid, pip_location)
    return generate_grids(mid_in_pips, grid_size, pip_location)

async def stream_price(client: OandaClient, instrument, grid):
    pricings = await client.stream_pricing(instrument)
    # logger.info()
    async for msg in pricings:
        if 'price' in msg:
            mid = get_mid(msg['price'])
            last_grid = max(x for x in grid.keys() if x < mid)
            next_grid = min(x for x in grid.keys() if x > mid)
            logger.info(f'last_grid: {mid - last_grid}, next_grid: {next_grid - mid}')

async def on_order_created(client: OandaClient, msg):
    pass
   
async def on_order_filled(client: OandaClient, msg):
    pass

async def on_order_cancelled(client: OandaClient, msg):
    pass

async def on_trade_closed(client: OandaClient, msg):
    pass
            
async def stream_transaction(client: OandaClient, grid):
    transactions = await client.stream_transactions()
    async for msg in transactions:
        if 'transaction' in msg:
            logger.info(msg['transaction'])
            

# Main asynchronous function
async def main(client):
    watchlist = await aprepare_watchlist(
        client,
        [
            'EUR', 'USD', 'JPY', 'AUD', 'GBP'
        ]
    )
    print(watchlist['EUR_USD'].pip_location)
    # return
    
    dataframes = await awarmup(
        client, watchlist, granularities=['D', 'H1']
    )
    
    selection_lst = list()
    
    key: str
    df: pd.DataFrame
    for key, df in dataframes.items():
        base, quote, tf = key.split('_')
        if tf != 'D':
            continue
        instr = f'{base}_{quote}'
        o, h, l, c = df.mid_o, df.mid_h, df.mid_l, df.mid_c
        df['atr'] = ta.ATR(h, l, c, timeperiod=14).mean()
        print(instr)
        selection_lst.append((
            key,
            price_to_pips(
                df['atr'].iloc[-1],
                watchlist[instr].pip_location
            )
        ))
    print(sorted(selection_lst, key=lambda x: x[1]))
    
    # grid = await generate_grid(client, instrument, 5)
    # await asyncio.gather(
    #     stream_price(client, instrument, grid),
    #     stream_transaction(client, grid)
    # )
            

# Entry point
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    client = get_client()
    
    try:
        loop.run_until_complete(client.initialize())
        loop.run_until_complete(main(client))
    except Exception as e:
        logger.exception(e, exc_info=True)
    finally:
        loop.run_until_complete(client.close())
        loop.close()
