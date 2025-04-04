# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%writefile app.py

import pickle
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as np

# loading the trained model
# =============================================================================
# pickle_in = open('Models/dt.pkl', 'rb')
# dt = pickle.load(pickle_in)
# pickle_in = open('Models/rf.pkl', 'rb')
# rf = pickle.load(pickle_in)
# pickle_in = open('Models/knn.pkl', 'rb')
# knn = pickle.load(pickle_in)
# pickle_in = open('Models/svm.pkl', 'rb')
# svm = pickle.load(pickle_in)
# pickle_in = open('Models/xgb.pkl', 'rb')
# xgb = pickle.load(pickle_in)
# #gbr = joblib.load("gbr_model.pkl")
# pickle_in = open('Models/mlr.pkl', 'rb')
# mlr = pickle.load(pickle_in)
# #pickle_in = open('Models/gbr.pkl', 'rb')
# #gbr = pickle.load(pickle_in)
# pickle_in = open('Models/lgbm.pkl', 'rb')
# lgbm = pickle.load(pickle_in)
# =============================================================================

gbr = joblib.load("Models/gbrjob.pkl")
knn = joblib.load("Models/knnjob.pkl")
rf = joblib.load("Models/rfjob.pkl")
mlr = joblib.load("Models/mlrjob.pkl")
dt = joblib.load("Models/dtjob.pkl")
xgb = joblib.load("Models/xgbjob.pkl")
svm = joblib.load("Models/svmjob.pkl")
lgbm = joblib.load("Models/lgbmjob.pkl")
scaler_x=joblib.load("Models/scaler_x.pkl")
scaler_y=joblib.load("Models/scaler_y.pkl")
scaler_y_svm=joblib.load("Models/scaler_y_svm.pkl")
scaler_x_svm=joblib.load("Models/scaler_x_svm.pkl")
abench=joblib.load("Models/abench.pkl")


type_apt=['Single Family One Storey','Single Family Two Storey','Townhouse', 'Apartment']
location_wo=['GREY','OWEN_SOUND','BRUCE','DUFFERIN','ORANGEVILLE','WELLINGTON','HURON','PERTH','STRATFORD','WATERLOO','GUELPH','KITCHENER','HAMILTON','NIAGARA','ST_CATHARINES','BRANTFORD','BRANT','HALDAND','NORFOLK','ELGIN','OXFORD','MIDDLESEX','LONDON','ST_THOMAS','SARNIA','LAMBTON','CHATHAM-KENT','ESSEX','WINDSOR']
location_eo=['KAWARTHA_LAKES','HALIBURTON','PETERBBOROUGH','NORTHUMBERLAND','COBOURG','RENFREW','HASTINGS','PRINCE_EDWARD','BELLEVILLE','LENNOX','ADDINGTON','FRONTENAC','KINGSTON','RENFREW','LANARK','LEEDS','GRENVILLE','BROCKVILLE','OTTAWA','CORNWALL','STORMONT','DUNDAS','GLLENGARRY','PRESCOTT','RUSSEL']
location_co=['MUSKOKA','SIMCOE','PEEL','YORK','TORONTO','HALTON','DURHAM','COLLINGWOOD', 'ORILLIA','BARRIE','NEWMARKET','AURORA','PICKERING','OSHAWA','AJAX','MARKHAM','BOWMANVILLE','MISSISSAUGA','OAKVILLE','BURLINGTON']
location_no=['ALGOMA','SUDBURY','MANITOULIN','PARRY_SOUND','NIPISSING','KENORA','RAINY_RIVER','THUNDER_BAY','COCHRANE','TIMISKAMING']
location_dict={'Northern Ontario':location_no, 'Eastern Ontario':location_eo, 'Western Ontario':location_wo, 'Central Ontario': location_co}
#[]
#{}

fig, ax = plt.subplots(figsize=(10, 5))

region_m= {0: 'Central Ontario', 1: 'Eastern Ontario', 2: 'Northern Ontario', 3: 'Western Ontario'}
type_m = {0: 'Apartment', 1: 'Single Family One Storey', 2: 'Single Family Two Storey', 3: 'Townhouse'}
location_m= {0: 'BANCROFT_AND_AREA', 1: 'BARRIE_AND_DISTRICT', 2: 'BRANTFORD_REGION', 3: 'CAMBRIDGE', 4: 'GREATER_TORONTO', 5: 'GREY_BRUCE_OWEN_SOUND', 6: 'GUELPH_AND_DISTRICT', 7: 'HAMILTON_BURLINGTON', 8: 'HURON_PERTH', 9: 'KAWARTHA_LAKES', 10: 'KINGSTON_AND_AREA', 11: 'KITCHENER_WATERLOO', 12: 'LONDON_ST_THOMAS', 13: 'MISSISSAUGA', 14: 'NIAGARA_REGION', 15: 'NORTHUMBERLAND_HILLS', 16: 'NORTH_BAY', 17: 'OAKVILLE_MILTON', 18: 'OTTAWA', 19: 'PETERBOROUGH_AND_KAWARTHAS', 20: 'QUINTE_AND_DISTRICT', 21: 'RIDEAU_ST_LAWRENCE', 22: 'SAULT_STE_MARIE', 23: 'SIMCOE_AND_DISTRICT', 24: 'TILLSONBURG_DISTRICT', 25: 'WOODSTOCK_INGERSOLL'}

#@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(Region, Location_u, Type_u, year):
  #year=2025
  boc=4.00
  
  #print("Region"+str(Region)+" Location: "+str(Location_u)+" Model to deploy: "+str(algo))
  cols=['Year', 'BOC Interest Rate', 'Region_Central Ontario',
       'Region_Eastern Ontario', 'Region_Northern Ontario',
       'Region_Western Ontario', 'Type_Apartment',
       'Type_Single Family One Storey', 'Type_Single Family Two Storey',
       'Type_Townhouse', 'Location_BANCROFT_AND_AREA',
       'Location_BARRIE_AND_DISTRICT', 'Location_BRANTFORD_REGION',
       'Location_CAMBRIDGE', 'Location_GREATER_TORONTO',
       'Location_GREY_BRUCE_OWEN_SOUND', 'Location_GUELPH_AND_DISTRICT',
       'Location_HAMILTON_BURLINGTON', 'Location_HURON_PERTH',
       'Location_KAWARTHA_LAKES', 'Location_KINGSTON_AND_AREA',
       'Location_KITCHENER_WATERLOO', 'Location_LONDON_ST_THOMAS',
       'Location_MISSISSAUGA', 'Location_NIAGARA_REGION',
       'Location_NORTHUMBERLAND_HILLS', 'Location_NORTH_BAY',
       'Location_OAKVILLE_MILTON', 'Location_OTTAWA',
       'Location_PETERBOROUGH_AND_KAWARTHAS', 'Location_QUINTE_AND_DISTRICT',
       'Location_RIDEAU_ST_LAWRENCE', 'Location_SAULT_STE_MARIE',
       'Location_SIMCOE_AND_DISTRICT', 'Location_TILLSONBURG_DISTRICT',
       'Location_WOODSTOCK_INGERSOLL']
  cols_xgb=['Year', 'Region','Type','Location','BOC Interest Rate']
  pred_xgb= [[year,0,0,0, boc]]
  pred=[[year, boc,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]]
  pred_df=pd.DataFrame(pred, columns=cols)
  pred_df_xgb=pd.DataFrame(pred_xgb, columns=cols_xgb)
  b=0
  
  for _, p in abench.iterrows():
      if str(p['Location']).find(Location_u)!=-1 and str(p['Type'])==Type_u and int(p['Year'])==year:
          b=int(p['Benchmark price_SA'])
  
  
  #print(str(b))
  for k,v in region_m.items():
      if(str(v)==str(Region)):
          pred_df_xgb['Region']=pred_df_xgb['Region'].apply(lambda x:k)
          break
  for k,v in location_m.items():
      if(str(v).find(str(Location_u))!=-1):
          pred_df_xgb['Location']=pred_df_xgb['Location'].apply(lambda x:k)
          break
  for k,v in type_m.items():
    if(str(v)==str(Type_u)):
        pred_df_xgb['Type']=pred_df_xgb['Type'].apply(lambda x:k)
        break
  
  for c in pred_df.columns:
      if str(c).find(Type_u)!=-1:
          #print(c)
          pred_df[c]=pred_df[c].apply(lambda x:True)
          break
  
  for cl in pred_df.columns:
    if str(cl).find(Region)!=-1:
        #print(cl)
        pred_df[cl]=pred_df[cl].apply(lambda x:True)
    if str(cl).find(Location_u)!=-1:
        #print(cl)
        pred_df[cl]=pred_df[cl].apply(lambda x:True)
  
  #df_svm= scaler_x_svm.transform(pred_df_xgb)
  #svm_pred_scaled=svm.predict(df_svm)
  #svm_pred = scaler_y_svm.inverse_transform(svm_pred_scaled.reshape(-1, 1))
  
  df_knn = scaler_x.transform(pred_df)
  print(df_knn)
  knn_pred_scaled=knn.predict(df_knn)
  print(knn_pred_scaled)
  print("knn_pred_scaled shape:", knn_pred_scaled.shape)
  # Predict and reverse scale the values
  knn_pred = scaler_y.inverse_transform(knn_pred_scaled.reshape(-1, 1))
  
  r=[[int(rf.predict(pred_df)),int(dt.predict(pred_df)),int(mlr.predict(pred_df)),int(lgbm.predict(pred_df)),int(gbr.predict(pred_df)),int(xgb.predict(pred_df_xgb)),int(knn_pred)]]
  r_cols=['DT', 'RF','MLR','LGBM', 'GBR', 'XGB', 'KNN']
  r_df=pd.DataFrame(r, columns=r_cols)
  r_long = r_df.melt(var_name="Models", value_name="Estimated prices")

  
  #fig, ax = plt.subplots(figsize=(10, 5))  # Adjust size if needed
  sns.barplot(x="Models", y="Estimated prices", data=r_long, palette="viridis", ax=ax)

  
  for i, row in r_long.iterrows():
    ax.text(i, row["Estimated prices"] + 5000, f"{int(row['Estimated prices']):,}", 
            ha='center', va='bottom', fontsize=10, color='black')

 
  ax.set_title("Estimated "+str(year)+" Prices for "+str(Type_u)+" in "+str(Location_u)+" by Model", fontsize=14)
  ax.set_ylabel("Estimated Prices (CAD)", fontsize=12)
  ax.set_xlabel("Models", fontsize=12)
  ax.tick_params(axis='x', rotation=0)
  ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"CAD {int(x):,}"))

  
  #st.pyplot(fig)
  
  if year==2025:
      se="The actual predicted price of "
      ve=" is:\n\t "
  else:
      se="In "+str(year)+", the estimated price of "
      ve=" was:\n\t "
  
  txt=st.text(se+str(Type_u)+" in "+str(Location_u)+" area"+ve+str(int(rf.predict(pred_df)))+" when using Random forest model\n\t "+str(int(dt.predict(pred_df)))+" when using Decision Tree Regressor model\n\t "+str(int(mlr.predict(pred_df)))+" when using Multi Linear Regression model\n\t "+str(int(lgbm.predict(pred_df)))+" when using LGBM model\n\t "+str(int(gbr.predict(pred_df)))+" when using Gradient Boost Regressor model\n\t "+str(int(xgb.predict(pred_df_xgb)))+" when using XGBoost model\n\t "+str(int(knn_pred))+" when using KNN model\n ")
  ave=st.text("Average price in "+str(year)+" of "+str(Type_u)+" in "+str(Location_u)+" area is: "+str(b))
  return txt

# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:gray;padding:13px">
    <h1 style ="color:black;text-align:center;">Ontario House Price Prediction App</h1>
    </div>
    """

    # display the front end aspect
    Location_u=""
    st.markdown(html_temp, unsafe_allow_html = True)

    # following lines create boxes in which user can enter data required to make prediction
    year = st.selectbox('Year', ("2025","2024","2023","2022","2021","2020"))
    Type_u = st.selectbox('House Type',("Single Family One Storey","Single Family Two Storey","Townhouse","Apartment"))
    Region = st.selectbox('Region',("Nothern Ontario","Eastern Ontario", "Central Ontario","Western Ontario"))
    if (str(Region).find('Northern')!=-1):
      Location_u=st.selectbox('Location', ('ALGOMA','SUDBURY','MANITOULIN','PARRY_SOUND','NIPISSING','KENORA','RAINY_RIVER','THUNDER_BAY','COCHRANE','TIMISKAMING'))
    elif (str(Region).find('Eastern')!=-1):
      Location_u=st.selectbox('Location', ('KAWARTHA_LAKES','HALIBURTON','PETERBBOROUGH','NORTHUMBERLAND','COBOURG','RENFREW','HASTINGS','PRINCE_EDWARD','BELLEVILLE','LENNOX','ADDINGTON','FRONTENAC','KINGSTON','RENFREW','LANARK','LEEDS','GRENVILLE','BROCKVILLE','OTTAWA','CORNWALL','STORMONT','DUNDAS','GLLENGARRY','PRESCOTT','RUSSEL'))
    elif (str(Region).find('Central')!=-1):
      Location_u=st.selectbox('Location',('MUSKOKA','SIMCOE','PEEL','YORK','TORONTO','HALTON','DURHAM','COLLINGWOOD', 'ORILLIA','BARRIE','NEWMARKET','AURORA','PICKERING','OSHAWA','AJAX','MARKHAM','BOWMANVILLE','MISSISSAUGA','OAKVILLE','BURLINGTON'))
    elif (str(Region).find('Western')!=-1):
      Location_u=st.selectbox('Location',('GREY','OWEN_SOUND','BRUCE','DUFFERIN','ORANGEVILLE','WELLINGTON','HURON','PERTH','STRATFORD','WATERLOO','GUELPH','KITCHENER','HAMILTON','NIAGARA','ST_CATHARINES','BRANTFORD','BRANT','HALDAND','NORFOLK','ELGIN','OXFORD','MIDDLESEX','LONDON','ST_THOMAS','SARNIA','LAMBTON','CHATHAM-KENT','ESSEX','WINDSOR'))
    #ApplicantIncome = st.number_input("Applicants monthly income")
    #LoanAmount = st.number_input("Total loan amount")
    #Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    #result =""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        prediction(str(Region), str(Location_u), str(Type_u), int(year))
        #st.success(result)
        st.pyplot(fig)
        #print(result)

if __name__=='__main__':
    main()
  
      