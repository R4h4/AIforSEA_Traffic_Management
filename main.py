import json
import click

from preprocessing import Preprocessor
from predict import Predictor


def main(t_day, t_time):
    prep = Preprocessor(t_day, t_time)
    print("Loading and preprocessing data.")
    prep.load_data('training.csv')

    print("Initializing Predictor.")
    predictor = Predictor()
    predictor.predict(prep.values())
    predictor.save_predictions()

    with open('result.json', 'w') as fp:
        json.dump(predictor.predictions, fp)


if __name__ == '__main__':
    main(15, '5:0')
