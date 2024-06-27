
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Load the model
model_path = 'models/RiskPredictor_RF_V3.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Encoding Function
def encoding(item):
    if item in ['Pass', 'None']:
        return 1
    elif item in ['Minor', 'Fail', 'Within past year', 'Flagged']:
        return 2
    elif item in ['Within past 1-3 years', 'Major']:
        return 3
    elif item < 200:
        return 1
    elif 200 <= item <= 500:
        return 2
    else:
        return 3

# Prediction Function
def risk_predict(input_data):
    input_data = input_data.apply(lambda x: x.map(encoding))
    encoded_data = input_data.copy()
    cols = input_data.columns
    prediction = model.predict(input_data)
    prediction = pd.DataFrame(prediction, columns=['Prediction'])

    probability = model.predict_proba(input_data)
    column_titles = ['High %', 'Low %', 'Moderate %']
    pd.set_option('display.float_format', '{:.1f}'.format)
    pb_array = pd.DataFrame(probability * 100, columns=column_titles)
    pb_array = pb_array.round(1)

    features = model.feature_importances_
    feature_impo = pd.Series(features, index=cols)
    feature_impo = feature_impo.sort_values(ascending=False)

    return prediction, pb_array, feature_impo, encoded_data

# --------------- Streamlit Application ------------------------

# Add the "Make a Prediction" subheading
st.markdown("<h1 style='font-size:35px;text-align:center;'>Make a Prediction</h1>", unsafe_allow_html=True)

# Manually entering data
with st.expander("Enter Data Manually", expanded=True):
    st.markdown("<h1 style='font-size:20px;text-align:center;'>Select data from the dropdowns</h1>",
                unsafe_allow_html=True)

    clients = st.number_input("Number of clients served annually", min_value=0, step=1)
    infraction_type = st.selectbox("Infraction type", ["Major", "Minor", "None"])
    infraction_timeline = st.selectbox("Infraction timeline", ["Within past 1-3 years", "Within past year", "None"])
    public_complaints = st.selectbox("Public complaints", ["Major", "Minor", "None"])
    sentiment_analysis = st.selectbox("Sentiment analysis", ["Flagged", "None"])
    inspection_results = st.selectbox("Inspection results", ["Pass", "Fail", "None"])

    if st.button('Risk Analysis Result!', key='risk_analysis_button', help="Button to trigger risk analysis"):
        data = {'Annual Clients': [clients], 'Infraction Type': [infraction_type],
                'Infraction Timeline': [infraction_timeline], 'Public Complaints': [public_complaints],
                'Sentiment Analysis': [sentiment_analysis], 'Inspection Results': [inspection_results]}
        input_data = pd.DataFrame(data)
        st.write(input_data)
        predict, probability, feature, encoded_data = risk_predict(input_data)
        st.success("Risk Prediction & Probability:")
        concat_df = pd.concat([probability, predict], axis=1)
        st.write(concat_df)

        # Display feature importance bar chart
        st.markdown("<h2 style='font-size:18px;text-align:center;'>Feature Importance</h2>", unsafe_allow_html=True)
        st.bar_chart(feature)


# Add section to upload test dataset
st.markdown("<h1 style='font-size:35px;text-align:center;'>Upload Test Dataset</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    test_data = test_data.fillna('None')
    
    entities = test_data['Entity ID']
    test_data = test_data.drop(columns=['Entity ID','Risk Category','Total Risk Score'])

    # Initialize session state variables
    if 'show_feature_importance' not in st.session_state:
        st.session_state.show_feature_importance = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'probability' not in st.session_state:
        st.session_state.probability = None
    if 'encoded_data' not in st.session_state:
        st.session_state.encoded_data = None

    # Button to predict risk
    if st.button('Predict the Risk!'):
        st.session_state.predictions, st.session_state.probability, st.session_state.show_feature_importance, st.session_state.encoded_data = risk_predict(test_data)

    # Display predictions if available
    if st.session_state.predictions is not None:
        st.write(test_data)
        concatenated_df = pd.concat([entities, st.session_state.probability, st.session_state.predictions], axis=1)
        st.write(concatenated_df)

        col1, col2 = st.columns(2)

        # Fig1 ==> Feature importance
        with col1:
            fig1 = px.bar(x=st.session_state.show_feature_importance.index, y=st.session_state.show_feature_importance.values)
            fig1.update_layout(width=800, height=500, title="Feature Importance", title_x=0.35, title_font_size=24,
                                xaxis=dict(tickfont=dict(size=14), title=dict(text="")), yaxis=dict(title=dict(text="")))
            st.plotly_chart(fig1, use_container_width=True)

            # Fig2 ==> Risk Distribution
            fig2 = px.bar(x=concatenated_df['Prediction'].value_counts().index, y=concatenated_df['Prediction'].value_counts().values)
            fig2.update_traces(text=concatenated_df['Prediction'].value_counts().values, textposition="outside")
            fig2.update_layout(width=800, height=500, title="Prediction Result", title_x=0.35, title_font_size=24,
                                xaxis=dict(tickfont=dict(size=14), title=dict(text="")), yaxis=dict(title=dict(text="")))
            st.plotly_chart(fig2, use_container_width=True)

        # Fig3 ==> Distribution of Risk Score
        with col2:
            scored_data = pd.concat([entities, st.session_state.encoded_data], axis=1)
            scored_data['Score'] = scored_data.iloc[:, 1:].sum(axis=1)

            fig3 = px.scatter(scored_data, x='Score', title='Distribution of Risk Scores',
                                color='Score', hover_name='Entity ID', hover_data={'Entity ID': False, 'Score': True})
            fig3.update_layout(
                title_x=0.35, title_font_size=24, xaxis_title='Risk Score', width=800, height=500,
                yaxis=dict(showticklabels=False, tickvals=[], ticktext=[], title=''),
                xaxis=dict(tickfont=dict(size=14))
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Sidebar for entity selection
            st.sidebar.header("Choose your filter:")
            entity_list = st.sidebar.multiselect("Select the entity", entities)
            if entity_list:
                fig4 = go.Figure()
                for entity in entity_list:
                    entity_data = scored_data[scored_data['Entity ID'] == entity].squeeze()[1:]
                    entity_risk_level = concatenated_df[concatenated_df['Entity ID'] == entity]['Prediction'].values[0]
                    # st.write(entity_risk_level)
                    # st.write(entity_data[-1])
                    entity_legend_name = f"{entity} (Risk: {entity_risk_level}, Score: {entity_data[-1]})"
                    # st.write(entity_legend_name)
                    fig4.add_trace(go.Scatterpolar(r=entity_data[:-1].values, theta=entity_data[:-1].index, mode='markers',
                                                   fill='toself', name=entity_legend_name, showlegend=True))
                fig4.update_layout(
                    title='Entities Score Values', title_x=0.35, title_font_size=24,
                    polar=dict(radialaxis=dict(tickformat='d', dtick=1), angularaxis=dict(tickfont=dict(size=16))),
                    width=800, height=600, margin=dict(l=110, r=110),
                    legend=dict(font=dict(size=18), orientation='h'))
                fig4.update_yaxes(tickformat='d', dtick=1)
                fig4.update_xaxes(tickfont=dict(size=16))
                st.plotly_chart(fig4)
            else:
                st.write('Select an Entity ID')
else:
    st.write('Please upload a file')



