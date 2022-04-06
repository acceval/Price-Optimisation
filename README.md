[![Price Optimisation](https://github.com/acceval/Price-Optimisation/actions/workflows/main.yml/badge.svg)](https://github.com/acceval/Price-Optimisation/actions/workflows/main.yml)

# Price-Optimisation

## Feature Assesment

Return parameters that picked by users along with the p-values.

### Resource URL

```
https://price-segmentation.herokuapp.com/features_assessment
```

### Parameters

```
{"type":"B2B", "filepath" :"https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv", "features":["Recency", "Revenue_L12", "Customer_Size", "l3y_volume", "standard_cost", "l12_sales_vol", "Current_Price"], "price_feature":"Avg_Price_L3Y", "volume_feature":"l3y_volume", "product_feature":"Product_Group", "current_price":"Current_Price", "sales_volume":"l12_sales_vol", "standard_cost":"standard_cost", "segmentation_features":["Recency", "Revenue_L12", "Customer_Size"] }
```

1. type: Type of business
2. filepath: Path to csv file contains all the data

> Sample of the input file is [here](https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv). Please note that column names cannot contain space.

2. features: List of features

> List of features to be trained. Each feature should exist as a column name and cannot contain space. Currently, features cannot be date data.

3. price_feature: Mapping price feature
4. volume_feature: Mapping volume feature
5. product_feature: :Mapping product feature
6. current_price: Mapping current price feature
7. sales_volume: Mapping sales volume feature
8. standard_cost: Mapping standard cost feature
9. segmentation_features

> Feature that will be used as to do segmentation


### How To Call

```
curl -X POST -H 'Accept: application/json' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 497' -H 'Content-type: application/json' -H 'User-Agent: python-requests/2.27.1' -d '{"type":"B2B", "filepath" :"https://raw.githubusercontent.com/acceval/Price-Optimisation/main/B2B_clean.csv", "features":["Recency", "Revenue_L12", "Customer_Size", "l3y_volume", "standard_cost", "l12_sales_vol", "Current_Price"], "price_feature":"Avg_Price_L3Y", "volume_feature":"l3y_volume", "product_feature":"Product_Group", "current_price":"Current_Price", "sales_volume":"l12_sales_vol", "standard_cost":"standard_cost", "segmentation_features":["Recency", "Revenue_L12", "Customer_Size"] }' http://127.0.0.1:5050/price_optimisation

```

### Sample Output

Full sample output can be found at this [file](https://raw.githubusercontent.com/acceval/Price-Optimisation/main/output.json )

# Notes 

We are still considering, instead return all the records, should this model only return the parameters needed to perform the calculations by front end so the performance can be faster.
