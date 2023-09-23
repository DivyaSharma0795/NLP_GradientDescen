#  IDS 706 Week 2 assignment - testing our template using pandas 
import pandas as pd
from Pandas_Description import PandasDesc
from Pandas_Plot import PandasPlot
def test_Pandas():
    #   Reading Source Data from the Github Link
    DataSource_Link = "https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/Iris_Data.csv"
    data_s = pd.read_csv(DataSource_Link)
    df_s = pd.DataFrame(data_s)

    #   Creating the sample files in the Resources folder
    # Writing the summary statistics to a file Summary.md in output folder
    Desc_df = PandasDesc(df_s)
    with open("./Resources/Summary.md", "w", encoding="utf-8") as f:
        f.write(str(Desc_df))
    # Pasting the sample graph in the output folder
    PandasPlot(df_s)

    #   Reading the Reference Data from the Resources Folder
    DataReference_Link = "./Resources/Iris_Data.csv"
    data_r = pd.read_csv(DataReference_Link)
    df_r = pd.DataFrame(data_r)

    assert(PandasDesc(df_r).equals(df_s.describe()))


test_Pandas()
