# Loan Approval from Historical Data

## Background
Banks routinely lose money based on loans that eventually default. Per the Federal Reserve, at the height of the financial crisis in 2009-2010, the amount lost approached $500 billion. Most recently losses each quarter tend to approach $150 billion. Delinquency rates tend to be around 1.5% most recently. Because of this, it is vitally important for banks to ensure that they keep their delinquencies as low as possible.

### Main question: Can we accurately predict loan approval based on historical data?

#### Questions to guide assessing the main question:
1.	How can we confidentially determine whether a loan can be approved?
2.	What factors predict loan approval?
3.	Which variables best predict if a loan will be a loss, and how much is the average loss? 

#### Initial Source:  https://www.kaggle.com/wendykan/lending-club-loan-data
This data source was uploaded to kaggle.com from Lending Club, a peer to peer financial company. Essentially, people can go on to the website and ask for an unsecured loan of between $1,000 and $40,000. Other people go to the site and choose to invest in the loans. So, people are essentially lending to other people directly with Lending Club as a facilitator.

## Methods
- Preprocess data in Python and R
- Perform variable selection using different approaches
- Deal with class imbalance 
- Test variables selected using linear and non-linear ML algorithms 
- Tune hyperparameters of different algorithms to increase predictive performance
