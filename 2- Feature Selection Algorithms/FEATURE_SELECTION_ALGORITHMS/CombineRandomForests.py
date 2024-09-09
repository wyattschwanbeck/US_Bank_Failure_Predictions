
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
from boruta import BorutaPy

from CONFIG_FEATURE_SELECTION import is_within_6_months, FEATURE_OMISSION,\
 failed_banks, All_Files_Directory,close_date,ratio_features, OPTIONAL_FEATURE_OMISSION, RFB, merge_on_common_columns,RFE_RFR,RFB
 
merge_on_common_columns(RFB,RFE_RFR)


