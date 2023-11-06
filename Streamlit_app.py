import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import seaborn as sns
import us
from streamlit_option_menu import option_menu
import joblib
from scipy.stats import t

with st.sidebar:
    choose = option_menu("Wrist Circumference Analysis", ["EDA", "Prediction"],
                         icons=['graph-up', 'calculator'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "3!important", "background-color": "#c8e3d5"},
        "icon": {"color": "#d1829c", "font-size": "19px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#0c5b5b"},
    }
    )
######### EDA page #########   
if choose == "EDA":
    df_ansur2_all = pd.read_csv("Ansur2_all.csv", encoding='latin-1') 
    num_var = df_ansur2_all.columns.to_list()[1:94]
    reduced_df = df_ansur2_all[num_var]
    # Removing highly correlated features
    corr_matrix = reduced_df.corr().abs()
    high_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_mask = corr_matrix.mask(high_mask)
    drop_cols = [c for c in corr_matrix_mask.columns if any(corr_matrix_mask[c] > 0.90)]
    new_df = reduced_df.drop(drop_cols, axis=1)
    correlation_with_cir = new_df.corr()['wristcircumference'].abs()
    sorted_df = new_df[correlation_with_cir.sort_values(ascending=False).index]
    # Assuming sorted_df and sorted_correlation_matrix are defined previously

    ######### Heatmap Top N
    st.subheader("Interactive Correlation Heatmap")

    # Create a slider to select the number of top correlated variables
    selected_variables = st.slider("Select the Number of Top Correlated Variables", min_value=1, max_value=len(sorted_df.corr().columns) - 1, value=10, step=1)

    # Get top N correlated variables with 'wristcircumference'
    top_correlated_vars = sorted_df.corr()['wristcircumference'].abs().nlargest(selected_variables + 1).index

    # Filter the correlation matrix based on top N correlated variables
    filtered_correlation_matrix = sorted_df.corr().loc[top_correlated_vars, top_correlated_vars]

    # Create heatmap
    heatmap = px.imshow(filtered_correlation_matrix,
                    labels=dict(color="Absolute Correlation"),
                    x=filtered_correlation_matrix.columns,
                    y=filtered_correlation_matrix.columns,
                    color_continuous_scale='Mint')
    # Set the font size of x and y axis labels
    heatmap.update_xaxes(tickangle=45, tickmode='array', tickvals=list(range(len(filtered_correlation_matrix.columns))),
                    ticktext=filtered_correlation_matrix.columns, tickfont=dict(size=10))
    heatmap.update_yaxes(tickangle=45, tickmode='array', tickvals=list(range(len(filtered_correlation_matrix.columns))),
                    ticktext=filtered_correlation_matrix.columns, tickfont=dict(size=10))

    # Set the font size of the colorbar labels
    heatmap.update_coloraxes(colorbar_title_font=dict(size=12), colorbar_tickfont=dict(size=10))

    # Set the layout of the heatmap
    heatmap.update_layout(title=f'Top {selected_variables} Correlated Variables with Wrist Circumference',
                        width=800, height=800)

    # Display the heatmap
    st.plotly_chart(heatmap)
    
    
    vis_data = df_ansur2_all[['Gender', 'WritingPreference', 'BMI_Category', 'SubjectNumericRace','SubjectsBirthLocation','Branch','wristcircumference']]

    ######## map
    st.subheader('Average Wrist Circumference Based on Birth State')
    state_cir = vis_data[['SubjectsBirthLocation', 'wristcircumference']]
    # convert state to abbr state
    state_name_to_abbr = {state.name: state.abbr for state in us.states.STATES}
    state_cir['SubjectsBirthLocation'] = state_cir['SubjectsBirthLocation'].apply(lambda x: state_name_to_abbr.get(x, x))
    # calculate avg wir cir
    state_cir = state_cir.groupby('SubjectsBirthLocation')['wristcircumference'].mean().round(2).reset_index()
    state_name_to_abbr = {state.name: state.abbr for state in us.states.STATES}
    # removing other contries
    state_cir = state_cir[state_cir['SubjectsBirthLocation'].isin(state_name_to_abbr.values())]
    state_cir.rename(columns={'wristcircumference': 'Average_wristcircumference'}, inplace=True)

    # Create a choropleth map using Plotly Express
    fig_state_cir = px.choropleth(state_cir, 
                        locations='SubjectsBirthLocation', 
                        locationmode="USA-states",
                        color='Average_wristcircumference',
                        title='Birth Locations Count',
                        color_continuous_scale=px.colors.sequential.Mint) 
    # Set the map projection to display U.S. states
    fig_state_cir.update_layout(
        title_text = 'Average wrist circumference based on birth state',
        geo_scope='usa'
        # limite map scope to USA
    )

    # Show the map
    st.plotly_chart(fig_state_cir)

    ######### Variable: Gender, BMI, Writing Preferences, Races

    # Select box for choosing grouping variable
    st.subheader("Combined Statistical Representations")
    group_var = st.selectbox("Choose Grouping Variable", ['Gender', 'Writing Preference', 'BMI Category', 'Races'])
    # Define color variable based on the chosen group_var
    color_var = None
    if group_var == 'Gender':
        color_var = 'Gender'
        fig = px.histogram(vis_data, x="wristcircumference", color= 'Gender', marginal="box",
                    hover_data=vis_data.columns,opacity=0.9, histnorm='probability density',color_discrete_map={'Female': '#cf6478', 'Male': '#579a97'})

    if group_var == 'Writing Preference':
        color_var ='WritingPreference'
        fig = px.histogram(vis_data, x="wristcircumference", color= 'WritingPreference', marginal="box",
                    hover_data=vis_data.columns,opacity=0.9, histnorm='probability density',
                    color_discrete_map={'Right hand':'#92c7b9','Left hand': '#e47b84', 'Either hand (No preference)': '#f5d8db'})

    if group_var == 'BMI Category':
        color_var = 'BMI_Category'
        category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
        fig =  px.histogram(vis_data , title='Wrist Circumference Grouped by BMI', x="wristcircumference", color= 'BMI_Category', marginal="box",
                    hover_data=vis_data.columns,opacity=0.9, histnorm='probability density',  category_orders={'BMI_Category': category_order},
                    color_discrete_map={'Underweight':'#c8e3d5','Normal':'#579a97','Overweight': '#f49b87', 'Obese': '#cf6478'})

    if group_var == 'Races':
        race = {1:'White, not of Hispanic descent',2:'Black, not of Hispanic descent',3:'Hispanic',4:'Asian', 5:'Native American',
            6:'Native Hawaiian/Pacific Islander'} # other is 'More than one and Other'
        # convert number to race
        vis_data['SubjectNumericRace'] = vis_data['SubjectNumericRace'].apply(lambda x: 'More than one and Other' if x not in [1,2,3,4,5,6] else x)
        vis_data['SubjectNumericRace'] = vis_data['SubjectNumericRace'].apply(lambda x: race.get(x, x))
        category_order = ['White, not of Hispanic descent','Black, not of Hispanic descent', 'Hispanic', 'Asian','Native American','Native Hawaiian/Pacific Islander','More than one and Other']
        fig = px.histogram(vis_data, x="wristcircumference", color= 'SubjectNumericRace', marginal="box",
                    hover_data=vis_data.columns,opacity=0.7, histnorm='probability density',
                    category_orders={'SubjectNumericRace': category_order},
                    color_discrete_map={'White, not of Hispanic descent':'#f5d8db','Black, not of Hispanic descent':'#ad5e5f',
                                        'Hispanic': '#fdd0c2', 'Asian': '#f89a7d','Native American': '#d1829c',
                                        'Native Hawaiian/Pacific Islander':'#d53e4f','More than one and Other':'#747474'}
                    )

    # Set plot title and labels
    fig.update_layout(title=f"Combined Statistical Representations with Wrist Circumference - Grouped by {group_var}",
                    xaxis_title="Wrist Circumference")
    # Show the plot using Plotly in Streamlit
    st.plotly_chart(fig)


######### gender vs. branch && map
    st.subheader("Gender Distribution by Branch Category")
    ig_gender_branch = px.sunburst(vis_data, path=['Gender', 'Branch'],color_discrete_sequence=px.colors.diverging.Spectral)
    st.plotly_chart(ig_gender_branch, width=100)



######### Modeling page #########   
elif choose == "Prediction":
    # Load the trained model
    st.title('Wrist Circumference Prediction')
    model = joblib.load('model.pkl')
    
    race_cate = ['White, not of Hispanic descent','Black, not of Hispanic descent', 'Hispanic', 'Asian',
                 'Native American','Native Hawaiian/Pacific Islander','More than one and Other']
    race_map = {1:'White, not of Hispanic descent',2:'Black, not of Hispanic descent',3:'Hispanic',
                4:'Asian', 5:'Native American', 6:'Native Hawaiian/Pacific Islander'} # other is 'More than one and Other'
    
    # User input for features
    Shoe_Size = st.number_input('Shoe size US', min_value=3.5, step=0.5,format="%1f")
    SubjectNumericRace = st.selectbox('Race', race_cate) 
    Height = st.number_input('Height (inches)', min_value=1)
    Weight = st.number_input('Weight (lbs)', min_value=1)
    Age = st.number_input('Age', min_value=1)
    Gender = st.selectbox('Gender', ['Male', 'Female'])  # Use a selectbox for binary choice

    # Map gender to numerical values
    Gender = 1 if Gender == 'Male' else 0
    # Map race to numerical values
    SubjectNumericRace = [key for key, value in race_map.items() if value == SubjectNumericRace][0]
    if Gender == 1:
        footlength = 220+(Shoe_Size-3.5)*10
    else:
        footlength = 220+(Shoe_Size-5)*10



    # 95% interval
    df_ansur2_all = pd.read_csv("Ansur2_all.csv", encoding='latin-1') 

    confidence_level = 0.95
    degrees_of_freedom = 4  
    margin_of_error = t.ppf((1 + confidence_level) / 2, degrees_of_freedom) * np.std(df_ansur2_all['wristcircumference']) / np.sqrt(len(df_ansur2_all['wristcircumference']))

    # Predict wrist circumference when the user clicks the button
    if st.button('Predict Wrist Circumference'):
        input_data = {
            'Heightin': [Height],
            'Weightlbs': [Weight],
            'Age': [Age],
            'Gender': [Gender],
            'footlength': [footlength],
            'SubjectNumericRace':[SubjectNumericRace]

        }
    


        input_df = pd.DataFrame(input_data)
        input_df = pd.get_dummies(input_df, columns=['SubjectNumericRace'])
        columns_to_add = ['SubjectNumericRace_1', 'SubjectNumericRace_2', 'SubjectNumericRace_3',
                  'SubjectNumericRace_4', 'SubjectNumericRace_5', 'SubjectNumericRace_6',
                  'SubjectNumericRace_7']
        
        for column in columns_to_add:
            if column not in input_df.columns:
                input_df[column] = 0
        
        input_df = input_df[['Heightin','Weightlbs','Age','Gender','footlength']+columns_to_add]

        prediction = model.predict(input_df)[0]

        lower_bound = prediction - margin_of_error
        upper_bound = prediction + margin_of_error
    
        st.success(f'Predicted Wrist Circumference: {prediction:.2f} mm')
        st.success(f'95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})')
