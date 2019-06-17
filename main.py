import click

from preprocessing import Preprocessor
from predict import Predictor


@click.command()
@click.option(
    "-f",
    "--filename",
    metavar="FILENAME",
    type=str,
    help="Path/Filename of the csv containing the data to make predictions from.",
)
@click.option(
    "-d",
    "--day",
    metavar="DAY",
    type=int,
    help="Day of T.",
)
@click.option(
    "-t",
    "--time",
    metavar="TIME",
    type=str,
    help="Time of T in the format %H:%M.",
)
def main(filename, day, time):
    prep = Preprocessor(day, time)
    print("Loading and preprocessing data.")
    prep.load_data(filename)

    print("Initializing Predictor.")
    predictor = Predictor()
    predictor.predict(prep.values())
    predictor.save_predictions()


if __name__ == '__main__':
    main()
