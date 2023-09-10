from nicegui import ui
from async_v20 import OandaClient

import pandas as pd

# Remove wb later on...
# Leave it here for now to avoid breaking the code
async def acycle(pairs, client: OandaClient, wb: None):
    account_res = await client.account_summary()
    pricing_res = await client.get_pricing(','.join(pairs.keys()))
    
    prices = pricing_res.get('prices', 200)
    
    pricings = list()
    
    for price in prices:
        pricings.append(
            [
                price.instrument,
                1 * 10 ** pairs[price.instrument].pip_location,
                price.quote_home_conversion_factors.positive_units,
                price.quote_home_conversion_factors.negative_units,
            ]
        )
    pricings = sorted(pricings, key=lambda p: p[0])
    df = pd.DataFrame(
        pricings,
        columns=['instrument', 'pipLocation', 'positiveFactor', 'negativeFactor']
    )
    df.set_index('instrument', inplace=True)
    
    account = account_res.get('account')
    # print(account)
    # sheet['A1'].value = df
    # sheet['A1'].options(pd.DataFrame, expand='table').value = df
    
    # sheet['H2'].value = account.nav