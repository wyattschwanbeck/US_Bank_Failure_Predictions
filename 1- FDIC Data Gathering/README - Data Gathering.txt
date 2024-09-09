The C# project ExtractFDICBankData was written to extract the totality of available bank reports between 1992 and 2023.
- Stored in 23 seperate files 100 Variables per file and associated bank CERT, REPDTE, and BKCLASS:
	Moved from ExtractFDICBankData\bin\Debug\TotalResultsX.csv to 'All_Data' folder
- Failed Bank List - Downloaded from FDIC https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/
- Extracted properties with formulas provides the list of available ratio features for feature selection algorithm testing
- Liquidated Banks - Banks that are liquidated are omitted from final classification predictions within the model