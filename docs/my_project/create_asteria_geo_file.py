import numpy as np
import pandas as pd
import fnmatch # for wildcard string searches

str_dc = 81 # string number with which deep core starts
str_gen2 = 1001 # same but for gen2

# read ppc geometry files in
df_in = pd.read_csv("./files/geo-f2k.gen2", sep = '\t+' , names = (np.arange(7)), engine="python")
df_in.columns = ["dom_id", "mb_id", "x", "y", "z", "str_num", "om_num"]
# rearrange columns in dictionary to be saved as dataframe

new_z = df_in["z"] + 1948.07 # changing from the surface frame to the IceCube (AMANDA) coordinate system
new_z = new_z.round(2) #round to two decimals

om_type = np.where(df_in["str_num"] < str_dc, "i3", "dc")
om_type = np.where(df_in["str_num"] < str_dc, om_type, "md")

dict = {"str_num" : df_in["str_num"], "om_num" : df_in["om_num"], "x" : df_in["x"], "y" : df_in["y"], "z" : new_z, "om_type" : om_type}
df_out  = pd.DataFrame(dict)

# save dataframe as txt file
df_out.to_csv("./files/Gen2_geometry_preliminary.txt", header=False, index=False, sep = "\t")#, float_format="%.2f")