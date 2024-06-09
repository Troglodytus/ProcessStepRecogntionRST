import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_rows', None)

class PDSF:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def copy_pdsf(self, equipment, destination):
        # Exclude these files
        excluded_files = ['PREFLUSH.pdsf', 'CDS1.pdsf', 'CDS2.pdsf', 'CDS3.pdsf']

        # Set source and subfolder paths
        pdsf_path = '\\\\vihsdv004.eu.infineon.com\\eqdatvi\\THIN\\SPINETCH_SEZ_RST_TRC'
        VIEWERPATH = 'VIEWERPATH'
        
        # Convert single string to list for consistent processing
        if isinstance(equipment, str):
            equipment = [equipment]

        # Loop over all equipment(s)
        for equip in equipment:
            file_path = os.path.join(pdsf_path, equip, VIEWERPATH)
            print("pdsf path: ", file_path)

            # Get all .pdsf files in the source directory
            files = [f for f in os.listdir(file_path) if f.endswith('.pdsf') and f not in excluded_files]

            # Sort files by modified time (newest first)
            files.sort(key=lambda f: os.path.getmtime(os.path.join(file_path, f)), reverse=True)

            # Copy the 5 newest files
            for file in files[:5]:
                shutil.copy(os.path.join(file_path, file), destination)
                print(f'Copied {file} to {destination}')




    def returnlogistics(self, logistics):
        # Keys to look for
        keys = ["START_TIME", "A_MES_ENTITY", "A_EPA", "A_MID", "A_SLOT", "A_WAFER_ID"]

        # Empty dictionary to hold results
        results = {}

        # Iterate over keys
        for key in keys:
            # Find row in logistics where column 1 is the current key
            row = logistics[logistics[0] == key]

            # If such a row exists, take corresponding value from column 2
            if not row.empty:
                # If the key is "START_TIME", extract the date and time
                if key == "START_TIME":
                    date_str = row[1].values[0]
                    time_str = row[2].values[0]
                    start_datetime_str = date_str + " " + time_str
                    results["START_TIME"] = datetime.strptime(start_datetime_str, "%m/%d/%Y %H:%M:%S")
                else:
                    results[key] = row[1].values[0]

        # Return the results as a tuple
        return tuple(results.get(key, None) for key in ["START_TIME", "A_MES_ENTITY", "A_EPA", "A_MID", "A_SLOT", "A_WAFER_ID"])



    def maketimetable(self, values_df, logistics_values):
        # Unpack logistics_values into separate variables
        START_DATETIME, A_MES_ENTITY, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID = logistics_values
        
        # Start with an empty DataFrame
        timetable = pd.DataFrame()

        # Add the timestamp column by adding the "Time" value from values_df to START_DATETIME
        #timetable['timestamp'] = pd.to_datetime(values_df.loc[:, 'Time']) + pd.Timedelta(START_DATETIME)
        timetable['timestamp'] = START_DATETIME + values_df['"Time"']
      

        # Add the logistics_values as separate columns
        timetable['A_MES_ENTITY'] = A_MES_ENTITY
        timetable['A_EPA'] = A_EPA
        timetable['LOT_ID'] = LOT_ID
        timetable['A_SLOT'] = A_SLOT
        timetable['A_WAFER_ID'] = A_WAFER_ID
        
        # Add "RecipeStep" column from values_df
        timetable['RecipeStep'] = values_df['"RecipeStep"']

        
        return timetable



    def open_file(self, file_name):
        # Initialize empty lists to hold our two types of data
        values_data = []
        logistics_data = []
        header_data = []

        # Check if file_name ends with ".pdsf"
        if file_name.endswith(".pdsf"):
            # Construct full file path
            file_path = os.path.join(self.dir_path, file_name)
            # Open the file and process it line by line
            with open(file_path, 'r') as file:
                lines = file.readlines()
                header_data.append(lines[6].strip().split())

                for line in lines[7:]:  # Skip the first 7 lines, start only reading values
                    columns = line.strip().split()
                    # Check if columns has any non-None values
                    if any(item is not None for item in columns):
                        if len(columns) >= 4:
                            values_data.append(columns)
                        else:
                            logistics_data.append(columns)

        # Convert the lists into DataFrames
        header_df = pd.DataFrame(header_data)
        header = header_df.iloc[0]

        values_df = pd.DataFrame(values_data)
        # Set first row as column names in values_df
        values_df.columns = header.tolist()
        #print("file: ", file_name, "data: ", values_df)

        values_df['"Time"'] = pd.to_timedelta(values_df['"Time"'].astype(float), unit='s')

        logistics_df = pd.DataFrame(logistics_data)

        #Find row in logistics_df where column 1 is "LOGISTICS_1"
        logistics_1_row = logistics_df[logistics_df[0] == 'LOGISTICS_1']

        #Extract corresponding value in column 2 and split it by ";"
        logistics_1_data = logistics_1_row[1].str.split(';', expand=True)

        #Convert the result into a DataFrame
        logistics_1_df = pd.DataFrame(logistics_1_data)
        
        #Now split each cell in logistics_1_df by "=" sign and create a new DataFrame "logistics_2"
        logistics_2 = logistics_1_df.stack().str.split('=', expand=True)

        #Reset logistic_2 index for proper concatenation
        logistics_2.reset_index(drop=True, inplace=True)

        #Concatenate logistics_df and logistics_2 DataFrames
        logistics = pd.concat([logistics_df, logistics_2], axis=0)

        # return the values_df DataFrame
        return values_df, logistics



    def process_pdsf(self, file_name):
        if file_name.endswith('.pdsf'):
            values_df, logistics = self.open_file(file_name)

            START_TIME, A_MES_ENTITY, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID = self.returnlogistics(logistics)
            timetable = self.maketimetable(values_df, (START_TIME, A_MES_ENTITY, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID))


            print("file: ", file_name)
            print(timetable)

            return timetable



####### Original Code for Reference ##### 
    # def process_pdsf(self, file_name):
    #     for file_name in os.listdir(self.dir_path):
    #         values_df, logistics_df = self.open_file(file_name)

    #         #print(values_df.get('"Time"'))
    #         #print("logistics_df: ", logistics_df)


    #         # Find row in logistics_df where column 1 is "LOGISTICS_1"
    #         logistics_1_row = logistics_df[logistics_df[0] == 'LOGISTICS_1']

    #         # Extract corresponding value in column 2 and split it by ";"
    #         logistics_1_data = logistics_1_row[1].str.split(';', expand=True)

    #         # Convert the result into a DataFrame
    #         logistics_1_df = pd.DataFrame(logistics_1_data)

    #         # Now split each cell in logistics_1_df by "=" sign and create a new DataFrame "logistics_2"
    #         logistics_2 = logistics_1_df.stack().str.split('=', expand=True)

    #         # Reset logistic_2 index for proper concatenation
    #         logistics_2.reset_index(drop=True, inplace=True)

    #         # Concatenate logistics_df and logistics_2 DataFrames
    #         logistics = pd.concat([logistics_df, logistics_2], axis=0)

    #         #Look for values in dataframe and return to variables
    #         START_TIME, A_MES_ENTITY, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID = self.returnlogistics(logistics)
    #         #make timetable
    #         timetable = self.maketimetable(values_df, (START_TIME, A_MES_ENTITY, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID))


    #         # Print the head of the DataFrames
    #         #print("Head of Values:")
    #         #print(values_df.head())
    #         #print("\n")
    #         #print("logistics:")
    #         #print(logistics)
    #         #print(START_TIME, A_EPA, LOT_ID, A_SLOT, A_WAFER_ID)
    #         print("file: ", file_name)
    #         print(timetable)

    #         return timetable
