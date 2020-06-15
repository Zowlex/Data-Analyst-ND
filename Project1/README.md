# Project: Explore weather trends

Created By: Fares Lassoued
Last Edited: Apr 03, 2020 11:37 AM
Property: Project outline
Tags: Data Analyst Nanodegree

## Summary

In this project, we're asked to analyze local and global temperature data and compare the temperature trends where I live (Gafsa, Tunisia) to overall global temperature trends.

## Steps

### Step1: Extracting the data from database

- Find the nearest city to me

Using the provided workspace, I started by finding the closest city to where I live using the next SQL query

    SELECT * 
    FROM city_list
    WHERE country = 'Tunisia'

[query result](https://www.notion.so/3125406c43c64ed99631d1140376a7e5)

The database contains only Tunisia's Capital city (Tunis) which is about 380 km from where I live.

- Query the average temperatures for Tunis and download the results (tunis_results.csv)

    -- Given that there is only one city for the country Tunisia in the database we- 
    -- can get the result with different queries 
    SELECT *
    FROM city_data
    WHERE country = 'Tunisia' and city = 'Tunis'

- Query the global temperatures and download the results (global_results.csv)

    SELECT *
    FROM global_data

### Open the csv in python using pandas library

I chose to open the csv files in python

    #read data
    local_data = pd.read_csv('data/tunis_results.csv')
    global_data = pd.read_csv('data/global_results.csv')

    local_data.head()

[Output](https://www.notion.so/5036193a138b44c29b24ba7121c7fec0)

There were 5 years with missing avg_temp so I decided to remove them as a data cleaning habit

    local_data.dropna(inplace=True)

## Calculate moving averages

for local data

    df = local_data['avg_temp']
    #5-year moving average
    local_data['mv_avg_temp'] = df.rolling(window=5).mean()
    
    #10-year moving average
    #local_data['mv_avg_temp'] = df.rolling(window=10).mean()
    
    #50-year moving average
    #local_data['mv_avg_temp'] = df.rolling(window=50).mean()
    local_data.head(10)

[Output](https://www.notion.so/11181585fff3433b92e17777739e77ad)

same for global data

    df = global_data['avg_temp']
    #5-year moving average
    global_data['mv_avg_temp'] = df.rolling(window=5).mean()
    
    #10-year moving average
    #global_data['mv_avg_temp'] = df.rolling(window=10).mean()
    
    #50-year moving average
    #global_data['mv_avg_temp'] = df.rolling(window=50).mean()

## Data visualization

    plt.figure(figsize=(20,10))
    sns.lineplot(x='year',y='mv_avg_temp', data=local_data)
    sns.lineplot(x='year',y='mv_avg_temp', data=global_data)
    plt.title('Weather trends for Tunis and globally using 10-year moving averages')
    plt.ylabel('Temperature in C°')
    plt.legend(['Tunis, Tunisia','Global'])
    plt.show()

![Project%20Explore%20weather%20trends/Untitled.png](Project%20Explore%20weather%20trends/Untitled.png)

![Project%20Explore%20weather%20trends/Untitled%201.png](Project%20Explore%20weather%20trends/Untitled%201.png)

![Project%20Explore%20weather%20trends/Untitled%202.png](Project%20Explore%20weather%20trends/Untitled%202.png)

### Observations:

1. Average temperatures in Tunis are 2.2 times higher than global temperatures given that our country's climate is mediterranean with warm/dry summers and cold/mild winters
2. The world seems to get hotter with each year which proves the term 'global warming' given that the average global temperature is 1.12 higher from the past two and half centuries.
3. The local and global temperatures seem to be strongly correlated (0.47), which is logical (unless we live on another planet!). Also, this means we can estimate local temperatures based on global ones.
    ```
    global_data.corrwith(local_data['avg_temp'])
    # > avg_temp       0.470518 
    ```

4. There was a noticeable global and local temperature drop around the 1800s, I googled it and it turned out to be the [Little Ice Age](https://en.wikipedia.org/wiki/Little_Ice_Age) period where the earth has seen a period of cooling due to various factors like cyclical lows in solar radiation, heightened volcanic activity, changes in the ocean circulation, variations in Earth's orbit, etc... I hope that this phenomenon repeats itself again.

## Conclusion

The global warming is real and it doesn't matter which city we live in since the global temperatures are rising and it is strongly correlated with local temperatures. One must think about his small actions that might have a butterfly effect and encourage eco-friendly products.
