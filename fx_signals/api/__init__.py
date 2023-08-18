import logging
import v20

import pandas as pd
from ratelimit import limits

logger = logging.getLogger(__name__)



@limits(calls=120, period=1)
def api_call(api: v20.Context, fn, *args, **kwargs):
    return fn(*args, **kwargs)

def candles_to_df(candles: v20.instrument.Candlestick):
    pass

def get_df(
    api: v20.Context,
    instrument: str,
    granularity: str,
    count: int = 500
) -> pd.DataFrame:
    logger.info(f"Getting {instrument} {granularity} data")
    res = api_call(
        api,
        api.instrument.candles,
        instrument=instrument,
        granularity=granularity,
        count=count
    )
    
    headers = [
        'time',
        'open',
        'high',
        'low',
        'close',
    ]
    
    data = []
    incomplete_candles = list()

    candle: v20.instrument.Candlestick
    for candle in res.get("candles", 200):
        if candle.complete:
            data.append(
                [
                    candle.time,
                    candle.mid.o,
                    candle.mid.h,
                    candle.mid.l,
                    candle.mid.c,
                ]
            )
        else:
            incomplete_candles.append(candle)
    
    df = pd.DataFrame(data, columns=headers)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df, incomplete_candles