import datetime as dt
import pandas as pd
import geohash
from tqdm import tqdm as tqdm


class Preprocessor:
    """
    Performs all needed transformations to the dataset to be able to run the model
    Input: Data frame
    """
    def __init__(self, t_day, t_time):
        # Load the data and already perform some processing
        # Note: we assume that the the day is continuous to the trainings set
        self.t = dt.datetime.strptime(t_time, '%H:%M') + dt.timedelta(days=int(t_day))
        self.start = self.t - dt.timedelta(minutes=(299 * 15))
        self.data = pd.DataFrame()

    def load_data(self, filename):
        date_parser = lambda x: pd.datetime.strptime(x, "%H:%M")
        self.data = (pd.read_csv(filename, parse_dates=['timestamp'], date_parser=date_parser)
                       .assign(timestamp=lambda x: (x['timestamp'] + pd.to_timedelta(x['day'].apply(int), unit='D')))
                       .pipe(self._drop_oos)
                       .pipe(self._fill_na_rows)
                       .assign(weekday=lambda x: (x['day'] % 7),
                               hour=lambda x: x['timestamp'].dt.hour)
                       .pipe(self._agg_demand)
                       .sort_values(by=['geohash6', 'timestamp'])
                       .drop_duplicates(subset=['geohash6', 'timestamp']))

    def values(self):
        return self.data[['geohash6', 'demand', 'day', 'hour', 'weekday', 'total_demand', 'area_demand']]

    def _drop_oos(self, df: pd.DataFrame):
        # Remove all rows later than t and earlier than t-300
        df = df.copy()
        return df[(df['timestamp'] <= self.t) & (df['timestamp'] >= self.start)]

    def _fill_na_rows(self, df):
        df = df.copy()
        # We can append the missing rows by comparing the existing ones for each
        # geo-location with a date_range
        missing_rows = list()
        # All existing datetimes for our timeframe
        all_dt = pd.date_range(start=self.start,
                               end=self.t,
                               freq='15min')

        print("Filling missing values.")
        for ghash in tqdm(df.geohash6.unique()):
            geo_dates = pd.DatetimeIndex(df[df.geohash6 == ghash].datetime)
            missing_dates = all_dt.difference(geo_dates)

            for date in missing_dates:
                day = (date - dt.datetime(1900, 1, 1, 0, 0)).days + 1
                missing_rows.append({'geohash6': ghash,
                                     'day': day,
                                     'timestamp': date,
                                     'demand': 0})

        # Combine the dataset with the missing rows
        return pd.concat([df, pd.DataFrame(missing_rows)], sort=False)

    def _agg_demand(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Get total demand for each time step
        df_total_demand = df.groupby('timestamp', as_index=False).demand.sum()\
                            .rename({'demand': 'total_demand'}, axis=1)
        df_agg_nearest_neigh = self._get_area_demand(df)

        df = df.merge(df_total_demand, how='left', on='timestamp')
        df = df.merge(df_agg_nearest_neigh, how='left', on=['geohash6', 'timestamp'])
        return df

    # Get the aggregated demand for the area around each area (including the location itself)
    def _get_area_demand(self, df):
        df_area_demand = []
        print("Aggregating demand.")
        for ghash in tqdm(df.geohash6.unique()):
            area_codes = geohash.expand(ghash)
            df_temp = (df[df.geohash6.isin(area_codes)].groupby('timestamp', as_index=False).demand.sum()
                       .rename({'demand': 'area_demand'}, axis=1)
                       .assign(geohash6=ghash))
            df_area_demand.append(df_temp)
        return pd.concat(df_area_demand, sort=False)
