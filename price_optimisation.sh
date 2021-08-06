#!/bin/bash

# source /home/ubuntu/scripts/religion/env/bin/activate
# cd /home/ubuntu/scripts/religion

# local
python main.py --env local --type B2c --filepath B2C_clean.csv --features Recency Revenue_L12 Customer_Size l3y_volume standard_cost l12_sales_vol Current_Price --price_feature Avg_Price_L3Y --volume_feature l3y_volume --product_feature Product_Group --current_price Current_Price --sales_volume l12_sales_vol --standard_cost standard_cost --segmentation_features Recency Revenue_L12 Customer_Size
# CurrentPrice, l12_sales_vol, StandardCost

# python main.py --filepath https://raw.githubusercontent.com/acceval/Price-Segmentation/main/sample_input_file.csv --features Customer_Type Customer_Industry Grade Country Destination_Port City_State Shipping_Condition Export/Domestic QUANTITY --target Price_Premium --index Index --price_per_segment https://raw.githubusercontent.com/acceval/Price-Segmentation/main/price_per_segment.json --price_threshold https://raw.githubusercontent.com/acceval/Price-Segmentation/main/sample_threshold.json --price_threshold_power_index https://raw.githubusercontent.com/acceval/Price-Segmentation/main/sample_threshold_with_power_index.json

