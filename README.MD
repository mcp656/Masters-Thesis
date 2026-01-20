Paper: Empirical Asset Pricing via Machine Learning: The Predictive Content of Insider Trading (Grade A+/12)

Abstract: 
\
Insider trading anomalies are well documented in low-dimensional linear models, but their impact
in high-dimensional machine learning is unclear. This thesis examines whether insider trading
adds predictive value in the high-dimensional framework of Gu, Kelly, and Xiu (2020). We
assess out-of-sample predictive performance for U.S. equities (2014–2021) using models ranging
from simple linear regression to deep neural networks. Our findings are threefold. First, our
results align with Gu, Kelly, and Xiu (2020), confirming that neural networks perform best.
Second, we find no systematic evidence that insider trading provides predictive value within
high-dimensional machine learning models. Third, we document heterogeneity across firm types.
Insider trading enhances simple linear predictability across all firm sizes, particularly for large
firms. In addition, among liquid stocks, insider trading adds predictive value in nonlinear models
that appear robust to transaction costs. Overall, our results suggest that while insider signals
contain information, this information is mostly redundant when conditioned on the high-dimensional
information set used in modern machine learning models.

Primary Litterature:
\
Gu, Kelly, and Xiu (2020). "Empirical Asset Pricing via Machine Learning."

How to run the code:
\
This repository contains the Python and R code used in our thesis.\
\
To run the code, you need both Python and R installed. Furthermore, you need access to CRSP. We recommend using the setup guides provided by tidy-finance.org:\
\
https://www.tidy-finance.org/python/setting-up-your-environment.html\
https://www.tidy-finance.org/r/setting-up-your-environment.html\
\
\
In general, the code should run on both newer and older versions of these environments. That said, future compatibility issues may arise, as pandas occasionally deprecates functions.\
\
In addition to the programming environments, access to CRSP data is required to run the code as intended. Please also note that some time conventions for stock data may differ between datasets.\
\
Finally, access to an HPC environment is recommended for running the machine learning models. We use uCloud, accessed via KU, with the following specifications:\
\
64 Intel cores\
384 GB RAM}
