
# coding: utf-8

# # Covid-19 Visualization
# * Data Cleaning : Preprocessing and standardization.
# * Data Summarization : Combine the daily vaccine information with the existing virus summary.
# * Summary Visualizations : Generate attractive bar plots for various summaries.
# * Global Statistics Visualization : Compare the daily new cases, active cases and deaths.

# In[ ]:


# Import the Libaries
import math 
import numpy as np
import pandas as pd 
import plotly.express as ex
import plotly.graph_objects as go 
import plotly.offline as pyo 
from datetime import datetime 
pyo.init_notebook_mode()


# In[ ]:


# Import the Dataset
df_vacc = pd.read_csv('C:\Users\amira\Data Science Club at WUT\Dataset\COVID-19 World Vaccination Progress/country_vaccinations.csv')
df_summary = pd.read_csv('C:\Users\amira\Data Science Club at WUT\Dataset\Covid-19 Global Dataset/worldometer_coronavirus_summary_data.csv')
df_daily = pd.read_csv('C:\Users\amira\Data Science Club at WUT\Dataset\Covid-19 Global Dataset/worldometer_coronavirus_daily_data.csv')


# In[ ]:


df_vacc.head()


# ### Data Cleaning
# 
# We have to make sure that all the countries names to be in the same name 

# #### Compare df_vacc.country & df_summary.country
# 
# Find the Countries in the Vaccination Data not in Summary Data 

# In[ ]:


[x for x in df_vacc.country.unique() if x not in df_summary.country.unique()]


# **Replace** 
# 
# 'Czechia' == "Czech Republic" 'Isle of Man' == "Isle Of Man" 'United Kingdom' == "UK" 'United States' == "USA" 'Northern Cyprus' == "Cyprus" 'Falkland Islands' == "Falkland Islands Malvinas"  'Hong Kong' == "China Hong Kong"  'Macao' == "China Macao Sar"  'Trinidad and Tobago' == "Trinidad And Tobago"  'Turks and Caicos Islands' == "Tucks And Caicos 
# 
# **DROP**
# 
# England,Guernsey,Jersey,Saint Halena, Wales Scotland Northern Ireland **(since they are part of the UK)**

# In[ ]:


#Replace 
df_vacc.country = df_vacc.country.replace().replace({
      'Czechia' : 'Czech Republic',
      'Falkland Islands' : 'Falkland Islands Malvinas',
      'Hong Kong' : 'China Hong Kong',
      'Isle of Man' : 'Isle Of Man',
      'Macao' : 'China Macao Sar',
      'Northern Cyprus' : 'Cyprus',
      'Northern Ireland' : 'Ireland',
      'Trinidad and Tobago' : 'Trinidad And Tobago',
      'Turks and Caicos Islands' : 'Turks And Caicos',
      'United Kingdom' : 'UK',
      'United States' : 'USA'     
})


# In[ ]:


# drop these since they are included in UK 
df_vacc = df_vacc[df_vacc.country.apply(lambda x: x not in ['England','Guernsey', 'Jersey', 'Saint Halena' 'Scotland', 'Wales', 'Northern Ireland'])]


# In[ ]:


# function to easily agrregate columns
def aggregate(df: pd.Series, agg_col: str) -> pd.DataFrame:
    
    data = df.groupby("country")[agg_col].max()
    data = pd.DataFrame(data)
    
    return data


# In[ ]:


# define the columns we want to summarize
cols_to_summarize = ['people_vaccinated', 
                     'people_vaccinated_per_hundred', 
                     'people_fully_vaccinated', 
                     'people_fully_vaccinated_per_hundred', 
                     'total_vaccinations_per_hundred', 
                     'total_vaccinations']

summary = df_summary.set_index("country")
vaccines = df_vacc[['country', 'vaccines']].drop_duplicates().set_index('country')
summary = summary.join(vaccines)

for col in cols_to_summarize:   
    summary = summary.join(aggregate(df_vacc, col))

summary['percentage_vaccinated'] = summary.total_vaccinations / summary.population * 100
summary['tested_positive'] = summary.total_confirmed / summary.total_tests * 100


# In[ ]:


df_summary = summary 


# In[ ]:


df_summary.to_csv("input")


# In[ ]:


df_summary2 = pd.read_csv('./input')


# In[ ]:


df_summary2.head()


# # Visualizations
# 
# Befor we make a graph, make a helper functions to make it easiler 

# In[ ]:


def get_multi_line_title(title:str, subtitle:str):
    return f"{title}<br><sub>{subtitle}</sub>"

def visualize_column(data: pd.DataFrame, xcolumn: str, ycolumn:str, title:str, colors:str, ylabel="Count", n=None):
    hovertemplate ='<br><b>%{x}</b>'+f'<br><b>{ylabel}: </b>'+'%{y}<br><extra></extra>'    
    data = data.sort_values(ycolumn, ascending=False).dropna(subset=[ycolumn])        
    
    if n is not None: 
        data = data.iloc[:n]
    else:
        n = ""
    fig = go.Figure(go.Bar(
                    hoverinfo='skip',
                     x=data[xcolumn], 
                     y=data[ycolumn], 
                     hovertemplate = hovertemplate,
                     marker=dict(
                         color = data[ycolumn],
                         colorscale=colors     ,
                        ),
                    ),
                )
    
    fig.update_layout(
        title=title,
        xaxis_title=f"Top {n} {xcolumn.title()}",
        yaxis_title=ylabel,
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x"
    )
    
    fig.show()


# ## People Vaccinated-Continent & Country 

# In[ ]:


title = get_multi_line_title("People Vaccinated", "Individuals who received the first dose of the vaccine")
visualize_column(summary.reset_index(), 'continent', "total_vaccinations", title, "burgyl")


# In[ ]:


title = get_multi_line_title("People Vaccinated", "Individuals who received the first dose of the vaccine")
visualize_column(summary.reset_index(), 'country', "total_vaccinations", title, "burgyl", n=30 )


# ## Percentage Vaccinated-Continent & Country 

# In[ ]:


title = get_multi_line_title("Percentage Vaccinated", "Percentage of the total population that have received the first dose")
visualize_column(summary.reset_index(), 'continent', "percentage_vaccinated", title, "emrld", "Percentage(%)")


# In[ ]:


title = get_multi_line_title("Percentage Vaccinated", "Percentage of the total population that have received the first dose")
visualize_column(summary.reset_index(), 'country', "percentage_vaccinated", title, "emrld", "Percentage(%)", n=30)


# # People Fully Vaccinated-Continent & Country 

# In[ ]:


title = get_multi_line_title("People Fully Vaccinated", "Individuals who have received all doses of the vaccine")
visualize_column(summary.reset_index(), 'continent', "people_fully_vaccinated", title, "Pinkyl")


# In[ ]:


title = get_multi_line_title("People Fully Vaccinated", "Individuals who have received all doses of the vaccine")
visualize_column(summary.reset_index(), 'country', "people_fully_vaccinated", title, "Pinkyl", n=30 )


# # Teseted Positive-Continent & Country 

# In[ ]:


title = get_multi_line_title("Tested Positive ", "Fraction of  people that tested positive among those that were tested")
visualize_column(summary.reset_index(), 'continent',"tested_positive", title, "blues", ylabel='Percentage' )


# In[ ]:


title = get_multi_line_title("Tested Positive ", "Fraction of  people that tested positive among those that were tested")
visualize_column(summary.reset_index(), 'country',"tested_positive", title, "blues", n=30, ylabel='Percentage' )


# # Vaccines in Use 

# In[ ]:


data = summary.dropna(subset=['vaccines'])
data = summary.groupby('vaccines')['total_vaccinations'].sum()
data = pd.DataFrame(data).reset_index()

title = get_multi_line_title("Vaccines In Use", "Popular Vaccine Combinations that are used around the globe")
visualize_column(data, 'vaccines',"total_vaccinations", title, "delta")


# # Serious_or_Critical

# In[ ]:


data = summary.dropna(subset=['serious_or_critical'])
data = data.reset_index()

title = get_multi_line_title("Serious or Critical Cases", "Number of people who are currently critically ill due to Covid-19")
visualize_column(data, 'country',"serious_or_critical", title, "turbid", n=20)

