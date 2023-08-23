import pandas as pd


class MetricTracker:
    def __init__(self, epoch_keys, iter_keys, writer=None):
        self.writer = writer
        self._df_epoch = pd.DataFrame(
            index=epoch_keys,
            columns=[
                'mean',
                'std'
            ]
        )
        self._df_iter = pd.DataFrame(
            index=iter_keys,
            columns=[
                'counts',
                'sum',
                'square_sum',
                'mean',
                'square_mean',
                'std'
            ]
        )
        self.reset()

    def reset(self):
        for col in self._df_epoch.columns:
            self._df_epoch[col].values[:] = 0
        for col in self._df_iter.columns:
            self._df_iter[col].values[:] = 0
        
    def epoch_update(self, key, value):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._df_epoch.at[key, 'mean'] = value

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._df_iter.at[key, 'counts'] += n
        self._df_iter.at[key, 'sum'] += value * n
        self._df_iter.at[key, 'square_sum'] += value * value * n
        self._df_iter.at[key, 'mean'] = self._df_iter.at[key, 'sum'] / self._df_iter.at[key, 'counts']
        self._df_iter.at[key, 'square_mean'] = self._df_iter.at[key, 'square_sum'] / self._df_iter.at[key, 'counts']
        self._df_iter.at[key, 'std'] = (self._df_iter.at[key, 'square_mean'] - self._df_iter.at[key, 'mean'] ** 2) ** 0.5

    def result(self):
        return pd.concat([self._df_epoch, self._df_iter.loc[:, ['mean', 'std']]]).to_dict('index')

