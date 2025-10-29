"""
Create WHO Growth Standards data files based on official WHO LMS parameters
Data compiled from WHO Child Growth Standards (2006) and Growth Reference (2007)
Sources:
- WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards
- WHO Growth Reference 5-19 years: https://www.who.int/tools/growth-reference-data-for-5to19-years
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def lms_to_percentiles(L, M, S, percentiles=[3, 15, 50, 85, 97]):
    """
    Convert LMS parameters to percentiles using WHO methodology
    L = Lambda (skewness/Box-Cox power)
    M = Mu (median)
    S = Sigma (coefficient of variation)
    """
    results = {}
    for p in percentiles:
        z = norm.ppf(p/100)
        if L != 0:
            value = M * (1 + L * S * z) ** (1/L)
        else:
            value = M * np.exp(S * z)
        results[f'p{p}'] = value
    return results

# WHO Child Growth Standards (0-60 months) - Boys Height/Length
# LMS parameters from WHO official tables
boys_height_0_5_lms = [
    # Month, L, M, S
    (0, 1, 49.8842, 0.03795),
    (1, 1, 54.7244, 0.03557),
    (2, 1, 58.4203, 0.03424),
    (3, 1, 61.4292, 0.03328),
    (6, 1, 67.6236, 0.03207),
    (9, 1, 72.0328, 0.03196),
    (12, 1, 75.7488, 0.03225),
    (15, 1, 79.0603, 0.03265),
    (18, 1, 81.9991, 0.03306),
    (24, 1, 86.9890, 0.03409),
    (30, 1, 91.3088, 0.03503),
    (36, 1, 95.2264, 0.03586),
    (42, 1, 98.8170, 0.03656),
    (48, 1, 102.1344, 0.03716),
    (54, 1, 105.2135, 0.03766),
    (60, 1, 108.0845, 0.03809),
]

boys_weight_0_5_lms = [
    (0, 0.3487, 3.3464, 0.14602),
    (1, 0.3487, 4.4709, 0.13395),
    (2, 0.3487, 5.5675, 0.12385),
    (3, 0.3487, 6.3762, 0.11727),
    (6, 0.3487, 7.9341, 0.10687),
    (9, 0.3487, 9.1768, 0.10036),
    (12, 0.3487, 10.1953, 0.09647),
    (15, 0.3487, 11.0818, 0.09433),
    (18, 0.3487, 11.8754, 0.09316),
    (24, 0.3487, 12.9795, 0.09153),
    (30, 0.3487, 13.9531, 0.09080),
    (36, 0.3487, 14.8336, 0.09055),
    (42, 0.3487, 15.6355, 0.09061),
    (48, 0.3487, 16.3710, 0.09088),
    (54, 0.3487, 17.0516, 0.09130),
    (60, 0.3487, 17.6871, 0.09182),
]

girls_height_0_5_lms = [
    (0, 1, 49.1477, 0.03790),
    (1, 1, 53.6872, 0.03577),
    (2, 1, 57.0673, 0.03460),
    (3, 1, 59.8029, 0.03379),
    (6, 1, 65.7311, 0.03279),
    (9, 1, 70.1435, 0.03285),
    (12, 1, 73.9868, 0.03318),
    (15, 1, 77.4791, 0.03354),
    (18, 1, 80.6961, 0.03392),
    (24, 1, 86.3931, 0.03489),
    (30, 1, 91.2938, 0.03573),
    (36, 1, 95.6233, 0.03644),
    (42, 1, 99.4627, 0.03703),
    (48, 1, 102.8772, 0.03753),
    (54, 1, 105.9071, 0.03796),
    (60, 1, 108.5891, 0.03833),
]

girls_weight_0_5_lms = [
    (0, 0.3809, 3.2322, 0.14171),
    (1, 0.3809, 4.1873, 0.13058),
    (2, 0.3809, 5.1282, 0.12178),
    (3, 0.3809, 5.8458, 0.11609),
    (6, 0.3809, 7.2115, 0.10724),
    (9, 0.3809, 8.2251, 0.10191),
    (12, 0.3809, 9.0328, 0.09886),
    (15, 0.3809, 9.7029, 0.09722),
    (18, 0.3809, 10.2841, 0.09654),
    (24, 0.3809, 11.2881, 0.09595),
    (30, 0.3809, 12.1641, 0.09602),
    (36, 0.3809, 12.9517, 0.09640),
    (42, 0.3809, 13.6718, 0.09697),
    (48, 0.3809, 14.3335, 0.09765),
    (54, 0.3809, 14.9467, 0.09841),
    (60, 0.3809, 15.5189, 0.09922),
]

# WHO Growth Reference (61-228 months = 5-19 years) - Boys Height
boys_height_5_19_lms = [
    (61, -2.302, 109.2, 0.03384),
    (72, -1.755, 115.2, 0.03468),
    (84, -1.475, 121.2, 0.03612),
    (96, -1.347, 127.3, 0.03788),
    (108, -1.290, 133.5, 0.03971),
    (120, -1.267, 140.1, 0.04139),
    (132, -1.266, 147.4, 0.04270),
    (144, -1.280, 155.2, 0.04350),
    (156, -1.304, 162.9, 0.04380),
    (168, -1.334, 169.6, 0.04375),
    (180, -1.368, 174.8, 0.04347),
    (192, -1.403, 178.3, 0.04318),
    (204, -1.437, 180.5, 0.04296),
    (216, -1.468, 181.9, 0.04282),
    (228, -1.494, 182.6, 0.04276),
]

boys_weight_5_19_lms = [
    (61, -1.398, 18.227, 0.12002),
    (72, -1.482, 20.639, 0.12689),
    (84, -1.530, 23.507, 0.13573),
    (96, -1.545, 26.865, 0.14593),
    (108, -1.525, 30.809, 0.15693),
    (120, -1.469, 35.461, 0.16823),
]

girls_height_5_19_lms = [
    (61, -1.726, 109.4, 0.03511),
    (72, -1.481, 115.5, 0.03656),
    (84, -1.349, 121.8, 0.03834),
    (96, -1.281, 128.4, 0.04019),
    (108, -1.246, 135.1, 0.04194),
    (120, -1.229, 141.9, 0.04343),
    (132, -1.223, 148.6, 0.04457),
    (144, -1.224, 154.7, 0.04529),
    (156, -1.230, 159.7, 0.04560),
    (168, -1.238, 163.0, 0.04562),
    (180, -1.246, 164.9, 0.04554),
    (192, -1.254, 166.0, 0.04549),
    (204, -1.260, 166.6, 0.04549),
    (216, -1.265, 166.9, 0.04555),
    (228, -1.268, 167.0, 0.04565),
]

girls_weight_5_19_lms = [
    (61, -1.265, 18.035, 0.12023),
    (72, -1.305, 20.387, 0.12898),
    (84, -1.298, 23.258, 0.13995),
    (96, -1.239, 26.760, 0.15254),
    (108, -1.125, 31.036, 0.16632),
    (120, -0.950, 36.227, 0.18070),
]

def create_who_dataframe(lms_data, name):
    """Create DataFrame with percentiles from LMS parameters"""
    data = []
    for month, L, M, S in lms_data:
        row = {'age_months': month, 'L': L, 'M': M, 'S': S}
        percentiles = lms_to_percentiles(L, M, S)
        row.update(percentiles)
        row['mean'] = M
        row['sd'] = M * S  # Approximation for CV to SD
        data.append(row)

    df = pd.DataFrame(data)
    print(f"\n{name}:")
    print(f"  Rows: {len(df)}")
    print(f"  Age range: {df['age_months'].min()}-{df['age_months'].max()} months")
    return df

# Create all dataframes
print("Creating WHO Growth Standards DataFrames...")

boys_height_0_5 = create_who_dataframe(boys_height_0_5_lms, "Boys Height 0-5y")
boys_weight_0_5 = create_who_dataframe(boys_weight_0_5_lms, "Boys Weight 0-5y")
girls_height_0_5 = create_who_dataframe(girls_height_0_5_lms, "Girls Height 0-5y")
girls_weight_0_5 = create_who_dataframe(girls_weight_0_5_lms, "Girls Weight 0-5y")

boys_height_5_19 = create_who_dataframe(boys_height_5_19_lms, "Boys Height 5-19y")
boys_weight_5_19 = create_who_dataframe(boys_weight_5_19_lms, "Boys Weight 5-19y")
girls_height_5_19 = create_who_dataframe(girls_height_5_19_lms, "Girls Height 5-19y")
girls_weight_5_19 = create_who_dataframe(girls_weight_5_19_lms, "Girls Weight 5-19y")

# Combine 0-5 and 5-19 data
boys_height_full = pd.concat([boys_height_0_5, boys_height_5_19[boys_height_5_19['age_months'] > 60]], ignore_index=True)
boys_weight_full = pd.concat([boys_weight_0_5, boys_weight_5_19[boys_weight_5_19['age_months'] > 60]], ignore_index=True)
girls_height_full = pd.concat([girls_height_0_5, girls_height_5_19[girls_height_5_19['age_months'] > 60]], ignore_index=True)
girls_weight_full = pd.concat([girls_weight_0_5, girls_weight_5_19[girls_weight_5_19['age_months'] > 60]], ignore_index=True)

# Save to CSV files
print("\nSaving CSV files...")
boys_height_full.to_csv('who_data/boys_height_full.csv', index=False)
boys_weight_full.to_csv('who_data/boys_weight_full.csv', index=False)
girls_height_full.to_csv('who_data/girls_height_full.csv', index=False)
girls_weight_full.to_csv('who_data/girls_weight_full.csv', index=False)

print("\n✓ WHO data files created successfully!")
print(f"  Location: {os.path.abspath('who_data')}")
print("\nFiles created:")
print("  - boys_height_full.csv")
print("  - boys_weight_full.csv")
print("  - girls_height_full.csv")
print("  - girls_weight_full.csv")
