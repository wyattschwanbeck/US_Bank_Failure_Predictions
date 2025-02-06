## US Bank Failure Prediction Summary

This was created December of 2023 and served as my final project for my Master's degree. The full pdf providing greater detail is included along with all source code dealing with data gathering from [SEC's EDGAR API](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) (C#),  feature selection, and machine learning models (Python). Over 2200 variables were batch analyzed using feature selection algorithms. Each set of variables were then trained with quarterly data from 2000 - 2022 based on the categorization of [failed banks](https://www.fdic.gov/bank-failures/failed-bank-list) with Y=1 if the reporting date is within 6 months of the reported failure date.

### Selected Features

The best performing feature selection algorithm turned out to be an ensemble dataset composed of 36 similarly selected ratio variables from Random Forest Boruta and Random Forest Regression. 


<table><tbody><tr><td style="height:15.0pt;width:111pt;"><a href="https://banks.data.fdic.gov/docs/risview_properties.yaml">FDIC Var Name</td><td style="width:410pt;">Description</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LNOT3T12/FTS/FI">LNOT3T12R</a></td><td>ALL OTHER LNS &amp; LS * 3-12 MONS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RBC1AAJ/RAT/FI">RBC1AAJ</td><td>LEVERAGE RATIO-PCA</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/ORERES/FTS/FI">ORER</td><td>OTHER REAL ESTATE OWNED RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NARERES/FTS/FI">NARERESR</td><td>NONACCRUAL-RE*1-4 FAMILY RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LNRESNCR/RAT/FI">LNRESNCR</td><td>LOAN LOSS RESERVE/N/C LOANS</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/VOLIAB/CDI/FI">VOLIABR</td><td>VOLATILE LIABILITIES RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/SCRDEBT/FTS/FI">SCRDEBTR</td><td>DEBT SECURITIES RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/ROA/RAT/FI">ROA</td><td>Return on assets (ROA)</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/INTINCY/RAT/FI">INTINCY</td><td>INTEREST INCOME TO EARNING ASSETS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RBCRWAJ/RAT/FI">RBCRWAJ</td><td>TOTAL RBC RATIO-PCA</td></tr><tr><td style="height:15.0pt;"><a href="https://banks.data.fdic.gov/docs/risview_properties.yaml">IDDEPINR</td><td><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/DEPINS/CDI/FI">DEPINS</a>/<a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/DEPBEFEX/FTS/FI">DEPBEFEX<a/></td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LNOT3LES/FTS/FI">LNOT3LESR</td><td>ALL OTHER LNS &amp; LS*3 MO OR LESS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="">IDT1RWAJR</td><td> <a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RBCT1J/CDI/FI">RBCT1J </a> / <a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RWAJ/CDI/FI">RWAJ</a></td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NTR/FTS/FI">NTRR</td><td>NONTRANSACTION-TOTAL RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LIAB/FTS/FI">LIABR</td><td>TOTAL LIABILITIES RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/ROEINJR/RAT/FI">ROEINJR</td><td>RETAINED EARNINGS/AVG BK EQUITY</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RB2LNRES/FTS/FI">RB2LNRESR</td><td>ALLOWANCE FOR L&amp;L IN TIER 2 RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NAASSET/FTS/FI">NAASSETR</td><td>NONACCRUAL-TOTAL ASSETS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LNATRESR/RAT/FI">LNATRESRR</td><td>ALLOW FOR LOANS + ALLOC TRN RISK RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/EQ/FTS/FI">EQR</td><td>EQUITY CAPITAL RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NCRER/RAT/FI">NCRER</td><td>N/C REAL ESTATE LNS/REAL ESTATE</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NTLNLSR/RAT/FI">NTLNLSR</td><td>NET CHARGE-OFFS/LOANS &amp; LEASES</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/EEFFR/RAT/FI">EEFFR</td><td>EFFICIENCY RATIO</td></tr><tr><td style="height:15.0pt;"><a href="">RBCT1JR</td><td><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/RBCT1J/CDI/FI"> RBCT1J / ASSETS</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/P3ASSET/FTS/FI">P3ASSETR</td><td>30-89 DAYS P/D TOTAL ASSETS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/EQUPTOT/FTS/FI">EQUPTOTR</td><td>UP-NET &amp; OTHER CAPITAL RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/ROAPTX/RAT/FI">ROAPTX</td><td>Pretax return on assets</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/CD3T12S/FTS/FI">CD3T12SR</td><td>TIME DEP $250,000 OR LESS REMAINING MATURITY OR REPRICING 3-12 MONTHS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NPERFV/RAT/FI">NPERFV</td><td>NONPERF ASSETS/TOTAL ASSETS</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/LNCI1/CDI/FI">LNCI1R</td><td>C&amp;I LOANS-UNDER-100K-$ RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/P3RER/RAT/FI">P3RER</td><td>30-89 DAYS P/D-REAL ESTATE LOANS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/DEPINS/CDI/FI">DEPINSR</td><td>ESTIMATED INSURED DEPOSITS RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/COREDEP/CDI/FI">COREDEPR</td><td>CORE DEPOSITS RATIO</td></tr><tr><td style="height:15.0pt;">IDLNCORR</td><td><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/DEPINS/CDI/FI">NET LOANS AND LEASES </a> TO <a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/COREDEP/CDI/FI"> CORE DEPOSITS RATIO </a></td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/DEPLGAMT/CDI/FI">DEPLGAMTR</td><td>AMT DEP ACC GREATER THAN $250,000 RATIO</td></tr><tr><td style="height:15.0pt;"><a href="https://www7.fdic.gov/DICT/app/templates/Index.html#!/Details/NARECNOT/FTS/FI">NARECNOTR</td><td>NONACCRUAL OTHER CONSTR &amp; LAND RATIO</td></tr></tbody></table>

### Selected Model and Results

The highest performing model within testing was the [probabilistic neural network (PNN)](https://keras.io/examples/keras_recipes/bayesian_neural_networks/) which produced estimate distributions with which standard deviation can be calculated along with mean predictions. This model type excelled with the unbalanced nature of the rate of banking failures against non-failing banks (10:1 in training).

Notably, [Republic First Bank dba Republic Bank](https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/republicbank.html) failed in April of 2024 and was correctly identified as likely to fail within this model at 75%, 83%, and 87% from 12/30/2022 to 06/30/2023. Further, [Citizens Bank](https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/citizensbank.html) was classified as likely to fail during this time at 88%-93% and did fail in November, 2023. 

<br>![Plot of Failed 2023-2024 banks](PlottedResults.png)

### On the Benefits of Bayesian Neural Networks
Model outputs are prediction distributions rather than single numerical outputs. This allows us to measure the models degree of certainty with regard to predictions via standard deviation. 
<br>![Standard Deviation by Prediction](PlottedResults-STD-Dev-By-Prediction.png)

Each batch of predictions mean and standard deviation follow a horseshoe pattern when plotted  with least predicted failing banks landing on the bottom left and highest predicted failing banks landing on the top left with lower standard deviations.  

### Future Development

There is opportunity to improve this models precision although it is imperative that recall remain maximized due to the critical nature of bank failures. The top ~20% of false-positive non-failing banks may be further scrutinized as some may truly also be operating on the brink of failure.
