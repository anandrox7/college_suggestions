from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

numerical = ['GPA_x', 'SAT_MATH', 'SAT_CRITICAL', 'SAT_WRITING','SAT_TOTAL',
             'ACT_MAIN','Class_Rank','Recommendations','Essay','Extracurricular','Work_Exp']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/suggest-colleges',methods=['POST'])
def suggest_colleges():
    if request.method=='POST':
        result=request.form
        #print(result['math'])
        total = result['math']+result['reading']+result['writing']
        #print('Upto here')
	test_sample = np.array([result['gpa'],result['math'],result['reading'],result['writing'],
                                total,result['act'],result['rank'],result['reco'],
                                result['essay'],result['extra'],result['workex']])

	test_sample = np.reshape(test_sample,(1,-1))
        
        pkl_file = open('./models/model_rf_v1.pk', 'rb')
        rfmodel = pickle.load(pkl_file)
        prediction = rfmodel.predict(test_sample)

        print(prediction)
        
        class1=[]
        class2=[]
        class3=[]

        college = pd.read_pickle('./models/college_data.pkl')
        sc = MinMaxScaler(feature_range=(0,1))
        sc.fit(college[numerical])
        
        college[numerical] = sc.transform(college[numerical])
        print(college.head())
        test_sample = sc.transform(test_sample)
        test_sample = np.clip(test_sample,0,1)
        
        print(test_sample)
        
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
            

        final_dict = {'Ambitious: ':class1,'Moderate: ':class2,'Safe: ':class3}
        
        return render_template('result.html',prediction=final_dict)

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
    for i in a:
        cl.append(str(df.iloc[i,0])+", "+str(df.iloc[i,3]))

    return cl
    
if __name__ == '__main__':
	app.run()

