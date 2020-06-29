"""Utils Functions and Constants

Here you can store your utils functions.
    - `compute_info_value` : compute information value of column against target
    - `GENDER_COLS_COUPLES` : couples of male/female column

"""
import pandas as pd
import numpy as np
import scipy.stats.stats as stats
import logging

logger = logging.getLogger(__name__)


MAPPING_CONSTELLATIONS = {
    "Andromeda Galaxy (M31)": "Andromeda",
    "Andromeda I": "Andromeda",
    "Andromeda II": "Andromeda",
    "Andromeda III": "Andromeda",
    "Andromeda IX": "Andromeda",
    "Andromeda V": "Andromeda",
    "Andromeda VIII": "Andromeda",
    "Andromeda X": "Andromeda",
    "Andromeda XI": "Andromeda",
    "Andromeda XII": "Andromeda",
    "Andromeda XIX[60]": "Andromeda",
    "Andromeda XV": "Andromeda",
    "Andromeda XVII": "Andromeda",
    "Andromeda XVIII[60]": "Andromeda",
    "Andromeda XX": "Andromeda",
    "Andromeda XXI[57]": "Andromeda",
    "Andromeda XXII[57]": "Andromeda",
    "Andromeda XXIII": "Andromeda",
    "Andromeda XXIV": "Andromeda",
    "Andromeda XXIX": "Andromeda",
    "Andromeda XXV": "Andromeda",
    "Andromeda XXVI": "Andromeda",
    "Andromeda XXVIII": "Andromeda",
    "Antlia 2": "Antlia",
    "Antlia B": "Antlia",
    "Antlia Dwarf": "Antlia",
    "Aquarius Dwarf Galaxy (DDO 210)": "Aquarius",
    "Aquarius II": "Aquarius",
    "Barnard's Galaxy (NGC 6822)": "Barnard",
    "Boötes I": "Boötes",
    "Boötes II": "Boötes",
    "Boötes III": "Boötes",
    "Boötes IV": "Boötes",
    "Camelopardalis B": "Camelopardalis",
    "Canes Venatici I Dwarf": "Canes",
    "Canes Venatici II Dwarf": "Canes",
    "Carina Dwarf (E206-G220)": "Carina",
    "Carina II": "Carina",
    "Carina III": "Carina",
    "Cas 1 (KK98 19)": "Cas",
    "Cassiopeia Dwarf (Cas dSph, Andromeda VII)": "Cassiopeia",
    "Cassiopeia II (Andromeda XXX)": "Cassiopeia",
    "Cassiopeia III (Andromeda XXXII)": "Cassiopeia",
    "Cetus Dwarf": "Cetus",
    "Cetus III": "Cetus",
    "Columba I": "Columba",
    "Coma Berenices Dwarf": "Coma",
    "Crater II": "Crater",
    "DDO 99 (UGC 6817)": "Draco",
    "Donatiello I": "Donatiello",
    "Draco Dwarf (DDO 208)": "Draco",
    "Draco II": "Draco",
    "Dwingeloo 1": "Dwingeloo",
    "Dwingeloo 2": "Dwingeloo",
    "Eridanus II": "Eridanus",
    "ESO 274-01[70]": "ESO",
    "ESO 294-010": "ESO",
    "ESO 321-014[70]": "ESO",
    "ESO 325-11": "ESO",
    "ESO 383-087 (ISG 39)": "ESO",
    "ESO 410-G005": "ESO",
    "ESO 540-030 (KDG 2)": "ESO",
    "ESO 540-032": "ESO",
    "FM2000 1": "FM2000",
    "Fornax Dwarf (E356-G04)": "Fornax",
    "GR 8 (DDO 155)": "GR",
    "Grus I": "Grus",
    "Grus II": "Grus",
    "Hercules Dwarf": "Hercules",
    "HIPASS J1247-77": "HIPASS",
    "HIZSS 003": "HIZSS 003",
    "Holmberg II (DDO 50, UGC 4305)": "Holmberg II (DDO 50, UGC 4305)",
    "Horologium I": "Horologium",
    "Horologium II": "Horologium",
    "Hydra II": "Hydra",
    "Hydrus I": "Hydrus",
    "IC 10 (UGC 192)": "IC",
    "IC 1613 (UGC 668)": "IC",
    "IC 3104": "IC",
    "IC 342": "IC",
    "IC 4662 (ESO 102-14)": "IC",
    "IC 5152": "IC",
    "Indus II": "Indus",
    "KK98 35": "KK98",
    "KK98 77": "KK99",
    "KKh 060": "KKh",
    "KKh 086": "KKh",
    "KKH 11 (ZOAG G135.74-04.53)": "KKh",
    "KKH 12": "KKh",
    "KKH 37 (Mai 16)": "KKh",
    "KKh 98": "KKh",
    "KKR 03 (KK98 230)": "KKR",
    "KKR 25": "KKR",
    "KKs 3": "KKs",
    "KUG 1210+301B (KK98 127)": "KUG 1210+301B (KK98 127)",
    "Lacerta I (Andromeda XXXI)": "Lacerta I (Andromeda XXXI)",
    "Large Magellanic Cloud (LMC)": "Large Magellanic Cloud (LMC)",
    "Leo A (Leo III, DDO 69)": "Leo",
    "Leo I Dwarf (DDO 74, UGC 5470)": "Leo",
    "Leo II Dwarf (Leo B, DDO 93)": "Leo",
    "Leo IV Dwarf": "Leo",
    "Leo P": "Leo",
    "Leo T Dwarf": "Leo",
    "Leo V Dwarf": "Leo",
    "M110 (NGC 205)": "M110",
    "M32 (NGC 221)": "M32",
    "Maffei 1": "Maffei",
    "Maffei 2": "Maffei",
    "MB 1 (KK98 21)": "MB",
    "MB 3": "MB",
    "NGC 147 (DDO 3)": "NGC",
    "NGC 1560": "NGC",
    "NGC 1569 (UGC 3056)": "NGC",
    "NGC 185": "NGC",
    "NGC 2366": "NGC",
    "NGC 2403": "NGC",
    "NGC 247": "NGC",
    "NGC 300": "NGC",
    "NGC 3109": "NGC",
    "NGC 3741": "NGC",
    "NGC 404": "NGC",
    "NGC 4163 (NGC 4167)": "NGC",
    "NGC 4214 (UGC 7278)": "NGC",
    "NGC 5102": "NGC",
    "NGC 5206": "NGC",
    "NGC 5237": "NGC",
    "NGC 5253": "NGC",
    "NGC 55": "NGC",
    "Pegasus Dwarf Irregular (DDO 216)": "Pegasus",
    "Pegasus Dwarf Sph (And VI)": "Pegasus",
    "Pegasus III": "Pegasus",
    "Perseus I (Andromeda XXXIII)": "Pegasus",
    "Phoenix Dwarf Galaxy (P 6830)": "Phoenix",
    "Phoenix II": "Phoenix",
    "Pictor II": "Pictor",
    "Pisces Dwarf": "Pisces",
    "Pisces I": "Pisces",
    "Pisces II": "Pisces",
    "Pisces III (Andromeda XIII)": "Pisces",
    "Pisces IV (Andromeda XIV)": "Pisces",
    "Pisces V (Andromeda XVI)": "Pisces",
    "Reticulum II": "Reticulum",
    "Reticulum III": "Reticulum",
    "Sagittarius Dwarf Irregular Galaxy (SagDIG)": "Sagittarius",
    "Sagittarius Dwarf Sphr SagDEG": "Sagittarius",
    "Sagittarius II": "Sagittarius",
    "Sculptor Dwarf (E351-G30)": "Sculptor Dwarf (E351-G30)",
    "Segue 1": "Segue",
    "Segue 2": "Segue",
    "Sextans A (92205, DDO 75)": "Sextans",
    "Sextans B (UGC 5373)": "Sextans",
    "Sextans Dwarf Sph": "Sextans",
    "Small Magellanic Cloud (SMC, NGC 292)": "Small Magellanic Cloud (SMC, NGC 292)",
    "Triangulum Galaxy (M33)": "Triangulum",
    "Triangulum II": "Triangulum",
    "Tucana Dwarf": "Tucana",
    "Tucana II": "Tucana",
    "Tucana III": "Tucana",
    "Tucana IV": "Tucana",
    "UGC 4483": "UGC",
    "UGC 4879 (VV124)[61]": "UGC",
    "UGC 7577 (DDO 125)": "UGC",
    "UGC 8508 (I Zw 060)": "UGC",
    "UGC 8651 (DDO 181)": "UGC",
    "UGC 8833": "UGC",
    "UGC 9128 (DDO 187)": "UGC",
    "UGC 9240 (DDO 190)": "UGC",
    "UGCA 105": "UGCA",
    "UGCA 133 (DDO 44)": "UGCA",
    "UGCA 15 (DDO 6)": "UGCA",
    "UGCA 276 (DDO 113)": "UGCA",
    "UGCA 292": "UGCA",
    "UGCA 438 (ESO 407-018)": "UGCA",
    "UGCA 86": "UGCA",
    "UGCA 92": "UGCA",
    "Ursa Major I Dwarf (UMa I dSph)": "Ursa",
    "Ursa Major II Dwarf": "Ursa",
    "Ursa Minor Dwarf": "Ursa",
    "Virgo I": "Virgo",
    "Willman 1": "Willman",
    "Wolf-Lundmark-Melotte (WLM, DDO 221)": "Wolf-Lundmark-Melotte (WLM, DDO 221)",
}

GENDER_COLS_COUPLES = [
    (
        "Estimated_gross_galactic_income_per_capita_female",
        "Estimated_gross_galactic_income_per_capita_male",
    ),
    (
        "Expected_years_of_education_female_galactic_years",
        "Expected_years_of_education_male_galactic_years",
    ),
    (
        "Expected_years_of_education_female_galactic_years",
        "Expected_years_of_education_male_galactic_years",
    ),
    (
        "Intergalactic_Development_Index_IDI_female",
        "Intergalactic_Development_Index_IDI_male",
    ),
    (
        "Intergalactic_Development_Index_IDI_female_Rank",
        "Intergalactic_Development_Index_IDI_male_Rank",
    ),
    (
        "Labour_force_participation_rate__ages_15_and_older_female",
        "Labour_force_participation_rate__ages_15_and_older_male",
    ),
    (
        "Labour_force_participation_rate__ages_15_and_older_female",
        "Labour_force_participation_rate__ages_15_and_older_male",
    ),
    (
        "Mean_years_of_education_female_galactic_years",
        "Mean_years_of_education_male_galactic_years",
    ),
    (
        "Population_with_at_least_some_secondary_education_female__ages_25_and_older",
        "Population_with_at_least_some_secondary_education_male__ages_25_and_older",
    ),
]


max_bin = 20
force_bin = 3


def mono_bin(Y, X, n=max_bin):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame(
                {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)}
            )
            d2 = d1.groupby("Bucket", as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception:
            n = n - 1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {
                "X": notmiss.X,
                "Y": notmiss.Y,
                "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True),
            }
        )
        d2 = d1.groupby("Bucket", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
        d3.DIST_EVENT / d3.DIST_NON_EVENT
    )
    d3["VAR_NAME"] = "VAR"
    d3 = d3[
        [
            "VAR_NAME",
            "MIN_VALUE",
            "MAX_VALUE",
            "COUNT",
            "EVENT",
            "EVENT_RATE",
            "NONEVENT",
            "NON_EVENT_RATE",
            "DIST_EVENT",
            "DIST_NON_EVENT",
            "WOE",
            "IV",
        ]
    ]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return d3


def char_bin(Y, X):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]
    df2 = notmiss.groupby("X", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
        d3.DIST_EVENT / d3.DIST_NON_EVENT
    )
    d3["VAR_NAME"] = "VAR"
    d3 = d3[
        [
            "VAR_NAME",
            "MIN_VALUE",
            "MAX_VALUE",
            "COUNT",
            "EVENT",
            "EVENT_RATE",
            "NONEVENT",
            "NON_EVENT_RATE",
            "DIST_EVENT",
            "DIST_NON_EVENT",
            "WOE",
            "IV",
        ]
    ]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return d3


def compute_info_value(df1, target):

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r"\((.*?)\).*$").search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({"IV": iv_df.groupby("VAR_NAME").IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)
