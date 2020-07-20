import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats

__all__ = ['LosgisticModel']

class LogisticResults():
    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.params = params
        self.model = model

    def predict(self, x):
        return self.predict_func(x)
    def inv_predict(self, y):
        return self.inv_predict_func(y)

    @property
    def parameters(self):
        return pd.Series(self.params, index=self.param_names)

    def summary(self):
        out = f'{self.model.model_type[0]}-parameter logistic model\nNumber of observations (x): {self.model.x.shape[0]}\nParameters:\n'
        out += self.model.model_str.format(*tuple(self.params))
        out += '\n'
        for k,v in self.diagnostics.items():
            out += f'\n{k}: {v}'
        return out
    @property
    def __str__(self):
        return self.summary()

class LogisticModel():
    def __init__(self, model_type='4pl'):
        self.model_type = model_type

        if model_type == '3pl':
            self.model_func = self._3pl_func
            self.inv_model_func = self._inv_3pl_func
            self.param_names = ['b', 'd', 'e']
        elif model_type == '4pl':
            self.model_func = self._4pl_func
            self.inv_model_func = self._inv_4pl_func
            self.param_names = ['b', 'c', 'd', 'e']
        elif model_type == '5pl':
            self.model_func = self._5pl_func
            self.inv_model_func = self._inv_5pl_func
            self.param_names = ['b', 'c', 'd', 'e', 'f']
            
        ms = ''
        for name in self.param_names:
            ms += '%s = {:1.3g}\n' % name
        self.model_str = ms

    
    def fit(self, x, y, start_params=None, **kwargs):
        if start_params is None:
            start_params = self.guess_start_params(x, y)
        else:
            if type(start_params) is pd.Series:
                start_params = np.array([start_params[k] for k in self.param_names])

        self.start_params = start_params
        self.x = x
        self.y = y
        params, cost, diagnostics, model_kwargs = self._fit(x, y, start_params, **kwargs)
        self.params = params
        self.diagnostics = diagnostics
        self.cost = cost
        self.model_kwargs = model_kwargs
        
        res = LogisticResults(self, params,
                                model_type=self.model_type,
                                predict_func=self._freeze_params(self.model_func, params),
                                inv_predict_func=self._freeze_params(self.inv_model_func, params),
                                diagnostics=diagnostics,
                                cost=cost,
                                param_names=self.param_names,
                                start_params=start_params,
                                model_kwargs=model_kwargs)
        self.result = res
        return res

    def predict(self, x, params=None):
        if params is None:
            params = self.params
        else:
            if type(params) is pd.Series:
                params = np.array([params[k] for k in self.param_names])
            self.params = params

        if params is None:
            raise ValueError('Model not fitted and no params specified.')
        return self.model_func(params, x)

    def inverse_predict(self, y, params=None):
        if params is None:
            params = self.params
        else:
            if type(params) is pd.Series:
                params = np.array([params[k] for k in self.param_names])
            self.params = params

        if params is None:
            raise ValueError('Model not fitted and no params specified.')
        return self.inv_model_func(params, y)

    def fit_predict(self, x, y, start_params=None, **kwargs):
        res = self.fit(x, y, start_params, **kwargs)
        return res.predict(x)

    def _fit(self, x, y, start_params, **kwargs):
        optim_res = optimize.least_squares(fun=self._resid_func,
                                            x0=start_params,
                                            args=(self.model_func, x, y),
                                            **kwargs)

        diagnostics = dict(status=optim_res.status,
                           message=optim_res.message,
                           success=optim_res.success,
                           nfev=optim_res.nfev)
        return optim_res.x, optim_res.cost, diagnostics, kwargs

    def guess_start_params(self, x, y):
        """Use 2 parameter linear regression to solve for start parameters, using max(x), min(x) and 1 for d, c, and f"""
        d = np.max(y)
        if self.model_type == '3pl':
            c = 0
        elif self.model_type in ['4pl', '5pl']:
            c = np.max([0, np.min(y)])
        ytmp = (d - y) / (y - c)
        xtmp = x.copy()
        ind = (ytmp > 0) & (xtmp > 0) & (ytmp < np.inf)
        # ytmp[ytmp<=0] = np.min(ytmp[ytmp>0]) / 2
        ytmp = np.log(ytmp[ind])
        
        # xtmp[xtmp<=0] = np.min(xtmp[xtmp>0]) / 2
        xtmp = np.log(xtmp[ind])
        slope, intercept, r_value, p_value, std_err = stats.linregress(xtmp, ytmp)
        b = slope
        e = np.exp(intercept / b)

        if self.model_type == '3pl':
            return np.array([b, d, e])
        elif self.model_type == '4pl':
            return np.array([b, c, d, e])
        elif self.model_type == '5pl':
            return np.array([b, c, d, e, 1])


    @staticmethod
    def _resid_func(params, F, x, y):
        return F(params, x) - y

    @staticmethod
    def _freeze_params(func, params):
        def frozen(x):
            return func(params, x)
        return frozen

    @property
    def parameters(self):
        return pd.Series(self.params, index=self.param_names)
    @staticmethod
    def _3pl_func(params, x):
        """b, d, e = params[0], params[1], params[2]"""
        return  params[1] / (1 + np.exp(params[0] * np.log(x) - params[0] * np.log(params[2])))
    @staticmethod
    def _4pl_func(params, x):
        """b, c, d, e = params[0], params[1], params[2], params[3]"""
        return  (params[2] - params[1]) / (1 + np.exp(params[0] * np.log(x) - params[0] * np.log(params[3]))) + params[1]
    @staticmethod
    def _5pl_func(params, x):
        """b, c, d, e, f = params[0], params[1], params[2], params[3], params[4]"""
        return  (params[2] - params[1]) / (1 + np.exp(params[0] * np.log(x) - params[0] * np.log(params[3])))**params[4] + params[1]
    @staticmethod
    def _inv_3pl_func(params, y):
        """b, d, e = params[0], params[1], params[2]"""
        return  np.exp(1 / params[0] * np.log( (params[1] / y) - 1 ) + np.log(params[2]))
    @staticmethod
    def _inv_4pl_func(params, y):
        """b, c, d, e = params[0], params[1], params[2], params[3]"""
        return  np.exp(1 / params[0] * np.log( ((params[2] - params[1]) / (y - params[1])) - 1 ) + np.log(params[3]))
    @staticmethod
    def _inv_5pl_func(params, y):
        """b, c, d, e, f = params[0], params[1], params[2], params[3], params[4]"""
        return  np.exp(1 / params[0] * np.log( ((params[2] - params[1]) / (y - params[1]))**(1/params[4]) - 1 ) + np.log(params[3]))

def test_models():
    """A data generating model that should be fit well by 3PL, 4PL and 5PL"""
    params_5pl = np.array([1, 0, 100, 10, 1])
    x = np.logspace(-2, 3, 15)
    y = LogisticModel._5pl_func(params_5pl, x) + np.random.randn(len(x)) * 2
    
    for i, t in enumerate(['3pl', '4pl', '5pl']):
        if t == '3pl':
            start_params = params_5pl[[0, 2, 3]]
        elif t == '4pl':
            start_params = params_5pl[[0, 1, 2, 3]]
        elif t == '5pl':
            start_params = params_5pl
        
        lm = LogisticModel(model_type=t)
        res = lm.fit(x, y, start_params=start_params)
        res = lm.fit(x, y)
        # print('start_params:', res.start_params)
        y_pred = lm.predict(x)
        y_pred_r = res.predict(x)
        np.testing.assert_allclose(y_pred, y_pred_r)
        y_pred = lm.fit_predict(x, y)

        y_pred = res.predict(x)
        x_inv = res.inv_predict(y_pred)
        np.testing.assert_allclose(x, x_inv)
        print(res.summary())
        print('\nAs series:')
        print(res.parameters, '\n')

        np.testing.assert_allclose(res.parameters.values, lm.parameters.values)
        np.testing.assert_allclose(res.params, lm.params)

        '''
        plt.figure(i + 1)
        plt.clf()
        axh = plt.subplot(111, xscale='log', yscale='linear')
        plt.scatter(x, y)
        x_new = np.logspace(-2, 3, 100)
        plt.plot(x_new, res.predict(x_new), '-')

    plt.figure(5)
    plt.clf()
    axh = plt.subplot(111, xscale='log', yscale='linear')
    for p in [[0.5, 0, 100, 10, 1],
              [1, 0, 100, 10, 1],
              [1, 20, 100, 10, 1],
              [1, 0, 100, 10, 0.5],
              [1, 0, 100, 10, 1]]:
        x = np.logspace(-2, 3, 50)
        y = LogisticModel._5pl_func(p, x) + np.random.randn(len(x)) * 0.000005
        plt.plot(x, y, '-o')
    '''
        

if __name__ == '__main__':
    test_models()

