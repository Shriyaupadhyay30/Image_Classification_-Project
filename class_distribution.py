#Function to create graphs with class distribution in dataset
def class_distribution(dataframe, col_name):

    #Making subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Percentage Plot', 'Total Count Plot'),
                        specs=[[{"type": "bar"}, {'type': 'scatter'}]])
    
    #Total counts in dataframe
    total_count = dataframe[col_name].value_counts().sum()
    #Percentage of particular label in dataframe
    percentage_values = (dataframe[col_name].value_counts().values / total_count) * 100

    #Creating bar plot
    fig.add_trace(go.Bar(y=percentage_values.tolist(),
                        x=[str(i) for i in dataframe[col_name].value_counts().index],
                        #Showing the values in percentage 
                        text=[f'{val:.2f}%' for val in percentage_values], 
                        textfont=dict(size=10),
                        name=col_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=colors)),
                                )
    
    #Creating scatter plot
    fig.add_trace(go.Scatter(x=dataframe[col_name].value_counts().keys(),
                         y=dataframe[col_name].value_counts().values,
                         mode='markers',
                         text=dataframe[col_name].value_counts().keys(),
                         textfont=dict(size=10),
                         marker=dict(size=15, color=colors),
                         name=col_name),
              row=1, col=2)

    #Updating plot
    fig.update_layout(title={'text': 'Disease Distribution in Dataset',
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')

    iplot(fig)
    

#Styling the plot with custom colours
colors = [
    '#3A506B', 
    '#8E8D8A',  
    '#D9BF77',  
    '#6A8D73',  
    '#B84A4A',  
    '#86B3D1',  
    '#B0C4B1',  
    '#9A5A6E',  
    '#C8A165',  
    '#7C6C8E'   
]

#Calling the function
class_distribution(df,'labels')