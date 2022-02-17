
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta, datetime 
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


  
 
LARGEFONT =("Verdana", 20)
country_dataset = pd.read_excel('EDGARv5.0_FT2019_fossil_CO2_booklet2020.xls',sheet_name='fossil_CO2_totals_by_country')

hawaii_df = pd.read_csv('archive.csv')

hawaii_df.dropna(inplace = True)

def convert_partial_year(number):
    year = int(number)
    d = timedelta(days=(number - year)*365)
    day_one = datetime(year,1,1)
    date = d + day_one
    return date
    

year_column = hawaii_df['Decimal Date']
years = []
for i in year_column:
  conv_year = convert_partial_year(i)
  years.append(conv_year)
  
hawaii_df['Decimal Date'] = years
hawaii_df['Decimal Date'] = pd.to_datetime(hawaii_df['Decimal Date']).dt.date

hawaii_df.set_axis(hawaii_df['Decimal Date'], inplace=True)
hawaii_df.drop(columns=['Year', 'Month', 'Seasonally Adjusted CO2 (ppm)', 
                        'Carbon Dioxide Fit (ppm)', 'Seasonally Adjusted CO2 Fit (ppm)'], inplace=True)


co2_data = hawaii_df['Carbon Dioxide (ppm)'].values
co2_data = co2_data.reshape((-1,1))



split_percent = 0.80
split = int(split_percent*len(co2_data))

co2_train = co2_data[:split]
co2_test = co2_data[split:]

date_train = hawaii_df['Decimal Date'][:split]
date_test = hawaii_df['Decimal Date'][split:]
   
look_back = 12
train_generator = TimeseriesGenerator(co2_train, co2_train, length=look_back, batch_size=2)     
test_generator = TimeseriesGenerator(co2_test, co2_test, length=look_back, batch_size=1)

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(look_back,1), return_sequences=True)) 
model.add(Dropout(0.05))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
num_epochs = 20
model.fit(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict(test_generator)
co2_train = co2_train.reshape((-1))
co2_test = co2_test.reshape((-1))
prediction = prediction.reshape((-1))

co2_data = co2_data.reshape((-1))




score = model.fit(train_generator, epochs=num_epochs, validation_data=test_generator, verbose=1)
losses = score.history['loss']
val_losses = score.history['val_loss']
plt.plot(losses, label='training')
plt.plot(val_losses, label='validation')
plt.title('Training loss vs Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['training','validation'], loc='best')
plt.show()


model.summary() 
        
def predict(num_prediction, model):
    prediction_list = co2_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list

def predict_dates(num_prediction):
    last_date = hawaii_df['Decimal Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq='M').tolist()
    return prediction_dates
    
num_prediction = 12
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)


plt.plot(forecast_dates, forecast)
plt.title('Predicted emissions')
plt.xlabel('Date')
plt.ylabel('CO2 Values (ppm)')
plt.show()

plt.plot(date_train,  co2_train)
plt.title('Hawaii Values')
plt.xlabel('Date')
plt.ylabel('CO2 Values (ppm)')
plt.show()

    
class tkinterApp(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
         
        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, CountryPage, PredictionPage):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  

  
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
         
        # label of frame Layout 2
        label = ttk.Label(self, text ="Start Page", font = LARGEFONT)
         
        # putting the grid in its place by using
        # grid
        label.grid(row = 0, column = 5, padx = 10, pady = 10)
  
        country_button = ttk.Button(self, text ="Country Emissions",
        command = lambda : controller.show_frame(CountryPage))
     
        # putting the button in its place by
        # using grid
        country_button.grid(row = 1, column = 4, padx = 10, pady = 10)
  
        ## button to show frame 2 with text layout2
        predict_button = ttk.Button(self, text ="Predicted Emissions",
        command = lambda : controller.show_frame(PredictionPage))
     
        # putting the button in its place by
        # using grid
        predict_button.grid(row = 1, column = 6, padx = 10, pady = 10)
  
          
  
  

class CountryPage(tk.Frame):
     
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Country Emissions", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 10)
        
        user_country = tk.Entry(self)
        user_country.grid(row = 1, column = 3)
  
        # button to show frame 2 with text
        # layout2
        search_button = ttk.Button(self, text ="Search",
                            command = lambda : show_emissions(country_dataset))
     
        # putting the button in its place
        # by using grid
        search_button.grid(row = 1, column = 4, padx = 10, pady = 10)
  
        # button to show frame 3 with text
        # layout3
        start_button = ttk.Button(self, text ="Start Page",
                            command = lambda : controller.show_frame(StartPage))
     
        # putting the button in its place by
        # using grid
        start_button.grid(row = 5, column = 3, padx = 10, pady = 10)
        
    
        def show_emissions(df):
            templist = []
            templist.append(user_country.get())
            searchdf = pd.DataFrame.melt(country_dataset, var_name='Year', value_name = 'Emission total', id_vars = ['country_name'])
            searchdf['country_name'] = searchdf['country_name'].astype(pd.CategoricalDtype(categories = templist,ordered = True))
            searchdf.dropna(inplace = True)
            
            figure = plt.Figure(figsize=(5,5), dpi = 100)
            ax = figure.add_subplot(111)
            ax.plot(searchdf['Year'], searchdf['Emission total'])
            
            canvas = FigureCanvasTkAgg(figure, self)
            canvas.draw()
            canvas.get_tk_widget().grid(row = 4, column = 2)
            

class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Predict Emissions", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 10)
        
        num_months = tk.Entry(self)
        num_months.grid(row = 1, column = 3)
  
        # button to show frame 2 with text
        # layout2
        submit_button = ttk.Button(self, text ="Submit",
                            command = lambda : plot_predictions(num_months))
     
        # putting the button in its place by
        # using grid
        submit_button.grid(row = 1, column = 4, padx = 10, pady = 10)
  
        # button to show frame 3 with text
        # layout3
        start_button = ttk.Button(self, text ="Start Page",
                            command = lambda : controller.show_frame(StartPage))
     
        # putting the button in its place by
        # using grid
        start_button.grid(row = 5, column = 3, padx = 10, pady = 10)
        
        def plot_predictions(num_months):
            num_months = int(num_months.get())
            forecast = predict(num_months, model)
            forecast_dates = predict_dates(num_months)
            
            figure = plt.Figure(figsize=(5,5), dpi = 100)
            ax = figure.add_subplot(111)
            ax.plot(forecast_dates, forecast)
            
            canvas = FigureCanvasTkAgg(figure, self)
            canvas.draw()
            canvas.get_tk_widget().grid(row = 4, column = 2)

    
  
app = tkinterApp()
app.mainloop()

