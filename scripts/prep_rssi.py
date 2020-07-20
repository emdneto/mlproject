from os import path, getcwd, listdir
import pandas as pd
import datetime
from pprint import pprint

class Prepare:

    
    def __init__(self):
        self.path = path.abspath(getcwd())
        self.datasetPath = path.join(self.path, 'data', 'dataset')
        self.datasetStaticPath = path.join(self.datasetPath, 'static')
        self.datasetPedestrianPath = path.join(self.datasetPath, 'pedestrian')
        self.modifiedPath = path.join(self.path, 'data', 'modified')
        self.peak = None

    def preparing(self):
        print(f'[*] Current dataset Path: {self.datasetPath}')
        print(f'[*] Current dataset Static Path: {self.datasetStaticPath}')
        print(f'[*] Current dataset Pedestrian Path: {self.datasetPedestrianPath}')
        print('--------------------------------------------------------------------')
        csvFiles = listdir(self.datasetStaticPath)
        for csvFile in csvFiles:
            filepath = path.join(self.datasetStaticPath, csvFile)
            col_names = ['Timestamp','Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 'NetworkMode', 'RSRP', 'RSRQ', 'SNR', 'CQI', 'DL_bitrate', 'UL_bitrate', 'State']
            dataset = pd.read_csv(filepath)
            #test = dataset.head(10)]
            data_top = dataset.head()
            for i, row in dataset.iterrows():
                

                ts = row['Timestamp']
                date_time_obj = datetime.datetime.strptime(ts, '%Y.%m.%d_%H.%M.%S')
                date = date_time_obj.date()
                time = date_time_obj.time()
                day_time = ''
                if time >= datetime.time(0) and time <= datetime.time(5,59):
                    day_time = 'Midnight'
                    #print('Midnight')
                
                if time >= datetime.time(6) and time <= datetime.time(11,59):
                    day_time = 'Morning'
                    #print('Morning')

                if time >= datetime.time(12) and time <= datetime.time(17,59):
                    day_time = 'Afternoon'
                    #print('Afternoon')
                
                if time >= datetime.time(18) and time <= datetime.time(23,59):
                    day_time = 'Night'
                    #print('Night')
                
                weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
                day = date.weekday()
                week_day = weekDays[day]                
                #month = date_time_obj.month()
                month = date.strftime("%B")
                dataset.loc[i, "WeekDay"] = week_day
                dataset.loc[i, "DayPeriod"] = day_time
                dataset.loc[i, "Month"] = month
                names = ['Longitude', 'Latitude', 'Speed', 'CellID', 
'RSRP', 'RSRQ', 'SNR', 'RSSI', 'DL_bitrate', 'UL_bitrate', 'NRxRSRP','NRxRSRQ', 'ServingCell_Lon', 'ServingCell_Lat', 
'ServingCell_Distance', 'CQI']
                for name in names:
                    if dataset.loc[i, name] == '-':
                        dataset.loc[i, name] = 'NaN'


                #dataset.loc[i, "DL_UL"] = row['DL_bitrate'] + row['UL_bitrate']
                

            dataset = dataset[['Timestamp','Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 'NetworkMode', 
'RSRP', 'RSRQ', 'SNR', 'CQI', 'DL_bitrate', 'UL_bitrate','State', 'NRxRSRP','NRxRSRQ', 'ServingCell_Lon', 'ServingCell_Lat', 
'ServingCell_Distance', 'WeekDay', 'DayPeriod', 'Month', 'RSSI']]
            dataset.to_csv(path.join(self.modifiedPath, 'static_rssi.csv'), mode='a', header=False)
            print(dataset.head(10))
            
p = Prepare()

p.preparing()
'''        
            self.peak = dataset['DL_UL'].max()

            for i, row in dataset.iterrows():
                dataset.loc[i, "BW_RANK"] = self.BWRank(row['DL_UL'])
            
            print(dataset.head(100))
                #week_day = date_time_obj.date().weekday()
                #print(week_day)
                #print('Date:', date_time_obj.date())
                #print('Time:', date_time_obj.time())
                #print('Date-time:', date_time_obj)
            break
    
    def BWRank(self, tp, noise):

        percent_20 = (self.peak * 20)/100
        percent_21 = (self.peak * 21)/100
        percent_40 = (self.peak * 40)/100
        percent_41 = (self.peak * 41)/100 
        percent_60 = (self.peak * 60)/100
        percent_61 = (self.peak * 61)/100
        percent_80 = (self.peak * 80)/100
        percent_81 = (self.peak * 81)/100
        
        if tp > 100 and tp <= percent_20:
            return '2'
        
        if tp >= percent_21 and tp <= percent_40:
            return '4'

        if tp >= percent_41 and tp <= percent_60:
            return '6'

        if tp >= percent_61 and tp <= percent_80:
            return '8'
        
        if tp >= percent_81:
            return '10'
        
        return '0'
        #if tp > 100 and tp <= ((self.peak * 20)/100)
        
 '''       


