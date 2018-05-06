from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

numerical = ['GPA_x', 'SAT_MATH', 'SAT_CRITICAL', 'SAT_WRITING','SAT_TOTAL',
             'ACT_MAIN','Class_Rank','Recommendations','Essay','Extracurricular','Work_Exp','Acceptance_Rates']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/suggest-colleges',methods=['POST'])
def suggest_colleges():
    if request.method=='POST':
        result=request.form
        #print(result['math'])
        total = float(result['math'])+float(result['reading'])+float(result['writing'])
        print(total)
        #print('Upto here')
	test_sample_pred = np.array([float(result['gpa']),float(result['math']),float(result['reading']),float(result['writing']),
                                total,float(result['act']),float(result['rank']),float(result['reco']),
                                float(result['essay']),float(result['extra']),float(result['workex'])])

	test_sample_pred = np.reshape(test_sample_pred,(1,-1))
        
        pkl_file = open('./models/model_rf_v1.pk', 'rb')
        rfmodel = pickle.load(pkl_file)
        prediction = rfmodel.predict(test_sample_pred)

        print(prediction)

        if prediction==5:
            ar= random.uniform(0.07,0.2)
        elif prediction==4:
            ar= random.uniform(0.2,0.4)
        elif prediction==3:
            ar= random.uniform(0.4,0.6)
        elif prediction==2:
            ar= random.uniform(0.6,0.8)
        else:
            ar= random.uniform(0.8,1)

        test_sample = np.array([float(result['gpa']),float(result['math']),float(result['reading']),float(result['writing']),
                                total,float(result['act']),float(result['rank']),float(result['reco']),
                                float(result['essay']),float(result['extra']),float(result['workex']),ar])
        test_sample = np.reshape(test_sample,(1,-1))
        class1=[]
        class2=[]
        class3=[]

        college = pd.read_pickle('./models/college_data.pkl')
        sc = MinMaxScaler(feature_range=(0,1))
        #sc.fit(college[numerical])
        
        #college[numerical] = sc.transform(college[numerical])
        print(college.head())
        print(test_sample)
        #test_sample = sc.transform(test_sample)
        print(test_sample)
        #test_sample = np.clip(test_sample,0,1)
        
        #print(test_sample)
        
        if prediction == 5:
            college_ambi = college[college['Label']==5]
            class1 = helper(college_ambi,test_sample)
            templ=[]
            for i in class1:
                templ.append(i.split(",")[0])
            
            college_mod = college[(college['Label']==5) & (~college['University Name'].isin(templ))]
            class2 = helper(college_mod,test_sample)
            templ=[]
            for i in class2:
                templ.append(i.split(",")[0])
            
            college_safe = college[college['Label']==4 | (((college['Label']==5)) \
                                                          & (~college['University Name'].isin(templ)))]
            class3 = helper(college_safe,test_sample)
            
        elif prediction == 4:
            college_ambi = college[college['Label']==5]
            class1 = helper(college_ambi,test_sample)
            
            college_mod = college[college['Label']==4]
            class2 = helper(college_mod,test_sample)
            templ=[]
            for i in class2:
                templ.append(i.split(",")[0])
                
            college_safe = college[college['Label']==3 | (((college['Label']==4)) \
                                                          & (~college['University Name'].isin(templ)))]
            class3 = helper(college_safe,test_sample)
            
        elif prediction == 3:
            college_ambi = college[college['Label']==4]
            class1 = helper(college_ambi,test_sample)
            
            college_mod = college[college['Label']==3]
            class2 = helper(college_mod,test_sample)
            templ=[]
            for i in class2:
                templ.append(i.split(",")[0])

            
            college_safe = college[college['Label']==2 | (((college['Label']==3)) \
                                                          & (~college['University Name'].isin(templ)))]
            class3 = helper(college_safe,test_sample)
            
        elif prediction == 2:
            college_ambi = college[college['Label']==3]
            class1 = helper(college_ambi,test_sample)
            
            college_mod = college[college['Label']==2]
            class2 = helper(college_mod,test_sample)
            templ=[]
            for i in class2:
                templ.append(i.split(",")[0])
            
            college_safe = college[college['Label']==1 | (((college['Label']==2)) \
                                                          & (~college['University Name'].isin(templ)))]
            class3 = helper(college_safe,test_sample)
            
        else:
            college_ambi = college[college['Label']==2]
            class1 = helper(college_ambi,test_sample)
            templ=[]
            for i in class1:
                templ.append(i.split(",")[0])

            college_mod = college[(college['Label']==2) & (~college['University Name'].isin(templ))]
            class2 = helper(college_mod,test_sample)
            
            college_safe = college[college['Label']==1]
            class3 = helper(college_safe,test_sample)
            

        #final_dict = {'Moderate: ':class2,'Ambitious: ':class1,'Safe: ':class3}
        
        return render_template('result.html',ambi=class1,mod=class2,safe=class3)

def helper(df,test):
    l=[]
    for i,r in df[numerical].iterrows():
        #print(cosine_similarity(np.array(r).reshape(1,-1),test)[0,0])
        #l.append(cosine_similarity(np.array(r).reshape(1,-1),test)[0,0])
        #print(euclidean_distances(np.array(r).reshape(1,-1),test)[0,0])
        l.append(euclidean_distances(np.array(r).reshape(1,-1),test)[0,0])

    l = np.array(l)
    a = l.argsort()[:5][::1]
    cl=[]
    #print("======================")
    #for i in a:
    #    print i,l[i]
    
    for i in a:
        cl.append(str(df.iloc[i,0])+", "+str(df.iloc[i,3]))

    return cl
    
if __name__ == '__main__':
	app.run()

