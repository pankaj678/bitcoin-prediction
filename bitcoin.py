import matplotlib.pyplot as plt
import math
import numpy as numpy
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from matplotlib import style
import pandas as pandas
from datetime import datetime
from datetime import timedelta
style.use('seaborn-talk')
k=0
import quandl
#max limit from quandl 50 req/day
data_frame = quandl.get('BCHARTS/KRAKENUSD')
data_frame.to_csv('btc_new.csv')
data_frame=pandas.read_csv('btc_new.csv',parse_dates=True,index_col=0)
data_frame=data_frame[['Open','High','Low','Close']]
forecast_col='Close'
data_frame.fillna(-9999,inplace=True)

days_inFuture=int(math.ceil(0.01*len(data_frame)))

data_frame['label']=data_frame[forecast_col].shift(-days_inFuture)

x=numpy.array(data_frame.drop(['label'],1))
x=preprocessing.scale(x)
x_lately=x[-days_inFuture:]
x=x[:-days_inFuture:]


data_frame.dropna(inplace=True)
y=numpy.array(data_frame['label'])
data_frame.dropna(inplace=True)

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)
clf_linear=LinearRegression(n_jobs=-1)
clf_linear.fit(x_train,y_train)
accuracy=clf_linear.score(x_test,y_test)

forecasted_data=clf_linear.predict(x_lately)
data_frame['future_price']=numpy.nan

last_date=datetime.now()-timedelta(days=0)
for data in forecasted_data:

    next_date = last_date + timedelta(days=k)
    k+=1
    print(next_date,"-->  $",data)

    data_frame.loc[next_date]=[numpy.nan for _ in range(len(data_frame.columns)-1)]+[data]

print("accuracy-->",accuracy*100,"%  number of days forecasting -->",days_inFuture)
data_frame['Close'].plot()
data_frame['future_price'].plot()
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.show()

