import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from datetime import datetime, timedelta

from src.config.crawler_config import INVESTING_MONTH_MATCHING

class DataUpdater(object) :

    def __init__(self, exchange_rate_list :list) :
        self._exchange_rate_list = exchange_rate_list
        self._from_api_to_origin_path = "src/AI/preprocessing/origins"
        self._columns = ["Date", "Price", "Open", "High", "Low", "Change %"]
        self._targets = [pd.read_csv(self._get_csv_path(exchange_rate)) for exchange_rate in self._exchange_rate_list]

        self._driver_options = webdriver.ChromeOptions()
        self._driver_options.add_argument("headless")
        self._driver_options.add_argument("--log-level=3")
        self._driver_options.add_argument("--disable-loging")
        self._driver = webdriver.Chrome(options = self._driver_options)

    def save(self) -> None :
        for i, exchange_rate in enumerate(self._exchange_rate_list) :
            self._targets[i].to_csv(self._get_csv_path(exchange_rate), index = False)
    
    def update(self) :
        for i, exchange_rate in enumerate(self._exchange_rate_list) :
            new_data = {'Date':[], 'Price':[], 'Open':[], 'High':[], 'Low':[], 'Change %':[]}
            self._driver.get(f"https://www.investing.com/currencies/{exchange_rate.replace('_','-').lower()}-historical-data")
            soup = BeautifulSoup(self._driver.page_source, "html.parser")
            rows = soup.find("div", {"class":"mt-6 flex flex-col items-start overflow-x-auto p-0 md:pl-1"}).find("tbody").find_all("tr")

            for row in rows[2:]:
                cols = row.find_all("td")
                new_data['Date'].append(self._convert_date(cols[0].find("time").text))
                new_data['Price'].append(cols[1].text.replace(',',''))
                new_data['Open'].append(cols[2].text.replace(',',''))
                new_data['High'].append(cols[3].text.replace(',',''))
                new_data['Low'].append(cols[4].text.replace(',',''))
                new_data['Change %'].append(self._convert_change(cols[6].text))

            for key in new_data.keys() :
                new_data[key].reverse()
            
            if exchange_rate == "JPY_KRW" :
                new_data = self._JPY_to_JPY100(pd.DataFrame(new_data))
            else : new_data = pd.DataFrame(new_data)

            point = new_data.index[(new_data['Date'] == self._targets[i].tail(n=1).iloc[0, 0])].array[0]
            self._simple_concat(i, self._targets[i], new_data[point+1:])

            print(exchange_rate, self._targets[i])

        return self

    def get_targets(self) -> pd.DataFrame :
        return self._targets

    def _simple_concat(
            self, index :int, target :pd.DataFrame, new_data :pd.DataFrame
    ) :
        self._targets[index] = pd.concat([
            target,
            new_data
        ])
        return self

    def _drop_vol(self, index :int) :
        self._targets[index] = self._targets[index].drop(columns="Vol.")
        return self

    def _get_rid_of_percentage(self, index :int) :
        self._targets[index]["Change %"] = self._targets[index]["Change %"].apply(lambda x: x[:-1]).astype({"Change %":"float64"})
        return self

    def _reverse_rows(self, index :int) :
        self._targets[index] = self._targets[index].loc[::-1].reset_index(drop=True)
        return self

    def _JPY_to_JPY100(self, new_data :pd.DataFrame) -> pd.DataFrame :
        for column in ["Price", "Open", "High", "Low"] :
            new_data[column] = new_data[column].astype(float)*100
        return new_data

    def _get_csv_path(self, exchange_rate :str) -> str :
        return f"{self._from_api_to_origin_path}/{exchange_rate}.csv"

    def _convert_date(self, date :str) -> str :
        return date.replace(date[:3], INVESTING_MONTH_MATCHING[date[:3]]).replace(',','').replace(' ','/')

    def _convert_change(self, change :str) -> str :
        return change[1:].replace('%','') if change[0] == '+' else change.replace('%','')



def convert_to_date_format(date :str) -> str :
    split_date = date.split("/")
    split_date.insert(0, split_date.pop(-1))
    return '-'.join(split_date)

def convert_to_csv_date_format(date :str) -> str :
    split_date = date.split("-")
    split_date.append(split_date.pop(0))
    return '/'.join(split_date)

def get_next_1yr() -> pd.DataFrame :
    new_data = {'Date':[]}
    today = datetime.now()

    for n in range(1, 365) :
        next_day = today + timedelta(days=n)
        if next_day.date().isoweekday() > 5 : continue

        new_data["Date"].append(convert_to_csv_date_format(next_day))
        if len(new_data["Date"]) == 260 : break
    
    return pd.DataFrame(new_data)