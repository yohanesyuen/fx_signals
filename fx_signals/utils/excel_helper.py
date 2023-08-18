from xlwings import Book
import pandas as pd

def cycle(pairs, api, args, wb: Book):
        sheet = wb.sheets['Sheet1']
        pricing_res = api.pricing.get(
            args.config.active_account,
            instruments=','.join(pairs.keys()),
        )
        pricing = pricing_res.get('prices', 200)

        pricings = list()
        
        account_res = api.account.summary(args.config.active_account)
        account = account_res.get('account')
        
        for p in pricing:
            pricings.append(
                [
                    p.instrument,
                    1 * 10 ** pairs[p.instrument].pipLocation,
                    p.quoteHomeConversionFactors.positiveUnits,
                    p.quoteHomeConversionFactors.negativeUnits,
                ]
            )
        
        pricings = sorted(pricings, key=lambda p: p[0])
        
        df = pd.DataFrame(
            pricings,
            columns=['instrument', 'pipLocation', 'positiveFactor', 'negativeFactor']
        )
        df.set_index('instrument', inplace=True)
        sheet['A1'].value = df
        sheet['A1'].options(pd.DataFrame, expand='table').value = df
        
        sheet['H2'].value = account.NAV
