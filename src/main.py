"""try some simple functions"""


def np_pd_main():
    """try numpy and pandas"""

    # import
    import numpy as np
    import pandas as pd
    from pandas import DataFrame, Series

    # func
    series = Series({"a": 1, "b": 2, "c": 3})
    print(series.index)


def transformers_main():
    """try transformers"""

    # import
    from transformers import pipeline

    # func
    print(pipeline("sentiment-analysis")("I love you"))


if __name__ == "__main__":
    np_pd_main()
    # transformers_main()
