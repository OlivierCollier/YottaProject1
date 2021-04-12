import numpy as np
import pandas as pd

import src.config.base as base

from ..cleaning import correct_wrong_entries

test_data = pd.DataFrame(
    columns=[
        "DATE",
        "AGE",
        "JOB_TYPE",
        "STATUS",
        "EDUCATION",
        "HAS_DEFAULT",
        "BALANCE",
        "HAS_HOUSING_LOAN",
        "HAS_PERSO_LOAN",
        "CONTACT",
        "DURATION_CONTACT",
        "NB_CONTACT",
        "NB_DAY_LAST_CONTACT",
        "NB_CONTACT_LAST_CAMPAIGN",
        "RESULT_LAST_CAMPAIGN",
        "SUBSCRIPTION",
    ],
    data=[
        [
            "2010-02-24",
            123,
            "Retraité",
            "Marié",
            "Primaire",
            "No",
            680,
            "No",
            "No",
            "Fixe",
            513,
            2,
            89,
            7,
            "Echec",
            "No",
        ]
    ],
)
test_wrong_entries = base.config_client_data.get("wrong_entries")


def test_correct_wrong_entries(data=test_data, wrong_entries=test_wrong_entries):
    cleaned_df = correct_wrong_entries(data, wrong_entries)
    assert np.isnan(cleaned_df.loc[0, "AGE"])
