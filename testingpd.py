import pandas as pd
 
# date string
d_string = "2023-09-17 14:30:00"
 
# Convert the string to datetime
dt_obj = pd.to_datetime(d_string)
 
print(dt_obj.hour)