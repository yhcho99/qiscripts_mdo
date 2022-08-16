'''
QRFT 모델 input 분석
'''

'''
BTM : Book to market
'''
def get_book_to_market(self,
                       ffill=True, ffill_limit=12, resample_freq='1M',
                       date_from='1986-12-31', date_to='9999-12-31', enable_masking=True, enable_fit_to_meta=True,
                       enable_spinner=True
                       ):
    """

    :param ffill:
    :param ffill_limit:
    :param resample_freq:
    :param date_from:
    :param date_to:
    :param enable_masking:
    :param enable_fit_to_meta:
    :param enable_spinner:
    :return:
    """
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    pstkrv = get_fundamental_data('pstkrv', name='pstkrv') # Preferred Stock - Redemption Value
    pstkl = get_fundamental_data('pstkl', name='pstkl')
    pstk = get_fundamental_data('pstk', name='pstk')
    seq = get_fundamental_data('seq', name='seq')
    txditc = get_fundamental_data('txditc', name='txditc')

    ps = pstkrv.copy()
    ps[:] = np.where(pstkrv.isnull(), pstkl, pstkrv)
    ps[:] = np.where(ps.isnull(), pstk, ps)
    ps[:] = np.where(ps.isnull(), 0, ps)
    txditc = txditc.fillna(0.)
    be = seq + txditc - ps
    # seq : Stockholders Equity
    # txditc  : Deferred Taxes and Investment Tax Credit
    # ps : redeemable prefered stock

    mv = self.core_api.get_monthly_market_value(date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)
    return inf_to_nan(be / mv)

'''
ram_12m_0m
'''

def get_ram(self, pi_or_ti, periods):  # neut
    if pi_or_ti == 'pi':
        ret_func = self.get_monthly_price_return
    else:
        ret_func = self.get_monthly_total_return

    mom = ret_func(periods, 0)
    vol = ret_func(1, 0).rolling(periods).std(ddof=0)

    factor = mom / vol

    return factor

'''
vol_3m : 3달 변동성 
'''

def get_monthly_volatility(self, rolling_window=36, date_from='1986-12-31', date_to='9999-12-31',
                           enable_masking=True, enable_fit_to_meta=True, enable_spinner=True):
    with Spinner(text='[Compustat API] : Calculating and loading volatility', enable_spinner=enable_spinner):
        stock_return = self.get_monthly_price_data(date_from=date_from, date_to=date_to, enable_masking=False,
                                                   enable_fit_to_meta=True, enable_spinner=False, _warn=False)
        stock_return: pd.DataFrame = stock_return / stock_return.shift(1) - 1.

        # volatility making start
        volatility = stock_return.rolling(rolling_window).std(ddof=0).iloc[rolling_window - 1:]
        return volatility

'''
res_mom_12m_1m_0m :
res_vol_6m_3m_0m :
'''

def _get_residual(self, add_const, rolling_window, use_recent_n, subtract_recent_n,
                  date_from='1986-12-31', date_to='9999-12-31'):
    assert use_recent_n > subtract_recent_n, "제외할 최근 데이터 개수는 사용할 전체 데이터 개수보다 적어야 함"
    factors = self.get_three_factor(date_from=date_from, date_to=date_to)
    X = factors[['Mkt-RF', 'SMB', 'HML']]
    rf = factors[['RF']]
    p = self.core_api.get_monthly_price_data(enable_masking=False, date_from=date_from, date_to=date_to, _warn=False)
    Y = p / p.shift(1) - 1
    co_index = X.index.intersection(Y.index)
    columns = Y.columns
    X = X.loc[co_index].values
    rf = rf.loc[co_index].values
    Y = Y.loc[co_index].values
    Y = Y - rf
    X = np.concatenate([X, np.ones([len(X), 1])], axis=1)

    res_mom, res_vol = self._rolling_linear_regression(X, Y,
                                                       add_const,
                                                       rolling_window, use_recent_n, subtract_recent_n)

    res_mom = pd.DataFrame(res_mom, index=co_index, columns=columns)
    res_vol = pd.DataFrame(res_vol, index=co_index, columns=columns)
    return res_mom, res_vol

'''
at
'''
def get_asset_turnover(self, ffill=True, ffill_limit=12, resample_freq='1M',
                       date_from='1986-12-31', date_to='9999-12-31',
                       enable_masking=True, enable_fit_to_meta=True,
                       enable_spinner=True):
    params = {
        'ffill': ffill,
        'ffill_limit': ffill_limit,
        'resample_freq': resample_freq,
        'date_from': date_from,
        'date_to': date_to,
        'enable_masking': enable_masking,
        'enable_fit_to_meta': enable_fit_to_meta,
        'enable_spinner': enable_spinner
    }
    get_fundamental_data = partial(self.core_api.get_fundamental_data, **params)
    sale = get_fundamental_data("sale")
    op_asset = self.get_operating_asset(**params)
    op_liab = self.get_operating_liabilities(**params)
    factor = (sale / (op_asset - op_liab))
    return factor

'''
gpa
'''


def get_gpa(self, as_trailing=False, ffill=True, ffill_limit=12, resample_freq='1M',
            date_from='1986-12-31', date_to='9999-12-31', enable_masking=True, enable_fit_to_meta=True,
            enable_spinner=True):
    """

    :param as_trailing:
    :param ffill:
    :param ffill_limit:
    :param resample_freq:
    :param date_from:
    :param date_to:
    :param enable_masking:
    :param enable_fit_to_meta:
    :param enable_spinner:
    :return:
    """
    if as_trailing:
        item_gp = 'saleq-cogsq'
        item_at = 'atq'
        filing_type = 'quarterly'
        quarterly_to_annual_trailing = True
    else:
        item_gp = 'sale-cogs'
        item_at = 'at'
        filing_type = 'annual'
        quarterly_to_annual_trailing = False

    gp = self.core_api.get_fundamental_data(item_gp, filing_type, quarterly_to_annual_trailing,
                                            'gp', True, ffill, ffill_limit, resample_freq,
                                            date_from, date_to, enable_masking,
                                            enable_fit_to_meta,
                                            enable_spinner=enable_spinner)
    at = self.core_api.get_fundamental_data(item_at, filing_type, quarterly_to_annual_trailing,
                                            'at', True, ffill, ffill_limit, resample_freq,
                                            date_from, date_to, enable_masking,
                                            enable_fit_to_meta,
                                            enable_spinner=enable_spinner)
    return inf_to_nan(gp / at)


'''
rev_surp
'''


def get_revenue_surprise(self, ffill=True, ffill_limit=12, resample_freq='1M',
                         date_from='1986-12-31', date_to='9999-12-31',
                         enable_masking=True, enable_fit_to_meta=True,
                         enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    saleq = get_fundamental_data('saleq', filing_type='quarterly')
    cshprq = get_fundamental_data('cshprq ', filing_type='quarterly')

    ajexq = self.core_api.get_adjustment_factor(filing_type='quarterly',
                                                date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)
    rps = saleq / (cshprq * ajexq)
    yoy = rps - rps.shift(12)
    rs = yoy / yoy.rolling(48).std()
    return rs

'''
cash_at
'''

def get_cash_to_asset(self, ffill=True, ffill_limit=12, resample_freq='1M',
                      date_from='1986-12-31', date_to='9999-12-31',
                      enable_masking=True, enable_fit_to_meta=True,
                      enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    cheq = get_fundamental_data("cheq", filing_type='quarterly')
    atq = get_fundamental_data("atq", filing_type='quarterly')
    factor = cheq / atq
    return inf_to_nan(factor)

# cheq : Cash and Short-Term Investments
# atq : Assets - Total

'''
op_lev
'''

def get_operating_leverage(self, ffill=True, ffill_limit=12, resample_freq='1M',
                           date_from='1986-12-31', date_to='9999-12-31',
                           enable_masking=True, enable_fit_to_meta=True,
                           enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    xsga = get_fundamental_data('xsga', name='xsga')
    cogs = get_fundamental_data('cogs', name='cogs')
    at = get_fundamental_data('at', name='at')

    factor = (xsga + cogs) / at
    return inf_to_nan(factor)

'''
roe
'''


def get_roe(self, ffill=True, ffill_limit=12, resample_freq='1M',
            date_from='1986-12-31', date_to='9999-12-31', enable_masking=True,
            enable_fit_to_meta=True,
            enable_spinner=True):
    """

    :param ffill:
    :param ffill_limit:
    :param resample_freq:
    :param date_from:
    :param date_to:
    :param enable_masking:
    :param enable_fit_to_meta:
    :param enable_spinner:
    :return:
    """
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    roe = get_fundamental_data('ni / seq', name='roe')
    return inf_to_nan(roe)


'''
std_u_e
'''


def get_standardized_unexpected_earnings(self, ffill=True, ffill_limit=12, resample_freq='1M',
                                         date_from='1986-12-31', date_to='9999-12-31',
                                         enable_masking=True, enable_fit_to_meta=True,
                                         enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)
    eps = get_fundamental_data('epspxq ', filing_type='quarterly')

    ajexq = self.core_api.get_adjustment_factor(filing_type='quarterly',
                                                date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)

    adj_eps = eps / ajexq
    yoy = adj_eps - adj_eps.shift(12)
    sue = (yoy - yoy.rolling(9999, min_periods=1).mean()) / yoy.rolling(9999, min_periods=1).std()
    return sue


'''
ret_noa
'''

def get_return_on_net_operating_asset(self, ffill=True, ffill_limit=12, resample_freq='1M',
                                      date_from='1986-12-31', date_to='9999-12-31',
                                      enable_masking=True, enable_fit_to_meta=True,
                                      enable_spinner=True):
    params = {
        'ffill': ffill,
        'ffill_limit': ffill_limit,
        'resample_freq': resample_freq,
        'date_from': date_from,
        'date_to': date_to,
        'enable_masking': enable_masking,
        'enable_fit_to_meta': enable_fit_to_meta,
        'enable_spinner': enable_spinner
    }
    get_fundamental_data = partial(self.core_api.get_fundamental_data, **params)
    oiadp = get_fundamental_data('oiadp')
    ona = self.get_operating_asset(**params) - self.get_operating_liabilities(**params)
    return oiadp / ona

'''
etm
'''


def get_earning_to_market(self, as_trailing=False, ffill=True, ffill_limit=12, resample_freq='1M',
                          date_from='1986-12-31', date_to='9999-12-31', enable_masking=True,
                          enable_fit_to_meta=True,
                          enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   quarterly_to_annual_trailing=as_trailing,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    if as_trailing:
        e = get_fundamental_data('ibq', name='ibq', filing_type='quarterly')
    else:
        e = get_fundamental_data('ib', name='ib', filing_type='annual')

    mv = self.core_api.get_monthly_market_value(date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)
    return inf_to_nan(e / mv)


'''
ia_mv
'''


def get_linear_assumed_intangible_asset_to_market_value(self, ffill=True, ffill_limit=12, resample_freq='1M',
                                                        date_from='1986-12-31', date_to='9999-12-31',
                                                        enable_masking=True, enable_fit_to_meta=True,
                                                        enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)
    xrd = get_fundamental_data("xrd").fillna(0.)
    xsga = get_fundamental_data("xsga")
    xad = get_fundamental_data("xad")

    mv = self.core_api.get_monthly_market_value(date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)

    rc = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
    xsga = xsga + 0.8 * xsga.shift(12) + 0.6 * xsga.shift(24) + 0.4 * xsga.shift(36) + 0.2 * xsga.shift(48)
    xad = xad + 0.8 * xad.shift(12) + 0.6 * xad.shift(24) + 0.4 * xad.shift(36) + 0.2 * xad.shift(48)
    rc = rc[rc >= 0]
    xsga = xsga[xsga >= 0]
    xad = xad[xad >= 0]
    factor = (rc + xsga * 0.8 + xad * 0.5) / mv
    return factor

'''
ae_m
'''


def get_advertising_expense_to_market(self, ffill=True, ffill_limit=12, resample_freq='1M',
                                      date_from='1986-12-31', date_to='9999-12-31',
                                      enable_masking=True, enable_fit_to_meta=True,
                                      enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)
    xad = get_fundamental_data('xad', name='xad')

    mv = self.core_api.get_monthly_market_value(date_from=date_from,
                                                date_to=date_to,
                                                enable_masking=enable_masking,
                                                enable_fit_to_meta=enable_fit_to_meta,
                                                enable_spinner=enable_spinner)
    factor = xad / mv
    factor = factor[factor > 0]
    return inf_to_nan(factor)

'''
ia_ta
'''


def get_linear_assumed_intangible_asset_to_total_asset(self, ffill=True, ffill_limit=12, resample_freq='1M',
                                                       date_from='1986-12-31', date_to='9999-12-31',
                                                       enable_masking=True, enable_fit_to_meta=True,
                                                       enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)
    xrd = get_fundamental_data("xrd").fillna(0.)
    xsga = get_fundamental_data("xsga")
    at = get_fundamental_data("at")
    rc = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
    xsga = xsga + 0.8 * xsga.shift(12) + 0.6 * xsga.shift(24) + 0.4 * xsga.shift(36) + 0.2 * xsga.shift(48)
    rc = rc[rc >= 0]
    xsga = xsga[xsga >= 0]
    factor = (rc + xsga * 0.2) / at
    return factor

'''
rc_a
'''


def get_rnd_capital_to_asset(self, ffill=True, ffill_limit=12, resample_freq='1M',
                             date_from='1986-12-31', date_to='9999-12-31',
                             enable_masking=True, enable_fit_to_meta=True,
                             enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    xrd = get_fundamental_data('xrd', name='xrd')
    at = get_fundamental_data('at', name='at')

    rnd_cap = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
    rc = rnd_cap[rnd_cap > 0]

    factor = rc / at
    return inf_to_nan(factor)


'''
r_s
'''


def get_rnd_to_sale(self, ffill=True, ffill_limit=12, resample_freq='1M',
                    date_from='1986-12-31', date_to='9999-12-31',
                    enable_masking=True, enable_fit_to_meta=True,
                    enable_spinner=True):
    get_fundamental_data = partial(self.core_api.get_fundamental_data,
                                   ffill=ffill, ffill_limit=ffill_limit, resample_freq=resample_freq,
                                   date_from=date_from,
                                   date_to=date_to,
                                   enable_masking=enable_masking,
                                   enable_fit_to_meta=enable_fit_to_meta,
                                   enable_spinner=enable_spinner)

    xrd = get_fundamental_data('xrd', name='xrd')
    sale = get_fundamental_data('sale', name='sale')
    xrd = xrd[xrd > 0.]

    factor = xrd / sale
    return inf_to_nan(factor)



'''
r_a
'''

