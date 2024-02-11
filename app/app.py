from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, render_template, request


def task1(data,data2):
    xtrain=data.iloc[:,:-1]
    ytrain=data.iloc[:,-1]
    xtest=data2.copy()
    #scaling
    scaler=StandardScaler()
    xtrain_sc=scaler.fit_transform(xtrain)
    xtest_sc=scaler.transform(xtest)
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=1.2, min_samples=5)

    # Fit DBSCAN to the data
    model = dbscan.fit(xtrain_sc)
    labels = model.labels_
    sample_cores = np.zeros_like(labels, dtype=bool)
    sample_cores[model.core_sample_indices_] = True
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    silhouette = silhouette_score(xtrain_sc, labels)
    return n_clusters,silhouette
def task2(data,data2):
    lr=LogisticRegression()
    x_train=data.iloc[:,:-1] #all columns except last column
    y_train=data.iloc[:,-1] #only last column
    x_test=data2
    scaler=StandardScaler()
    xtrain_sc=scaler.fit_transform(x_train)
    #transform only on test data
    xtest_sc=scaler.transform(x_test)
    lr.fit(xtrain_sc,y_train)
    y_train_pred=lr.predict(xtrain_sc)
    train_accuracy=accuracy_score(y_train,y_train_pred)
    y_pred=lr.predict(xtest_sc)
    return train_accuracy,y_pred
def task3(data):
    data.position=data.position.apply(lambda x:'inside' if x=='Inside' or x=='inside' else 'outside')
    data.date=data.date.astype(str)
    data.time=data.time.astype(str)
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.sort_values(by=['location', 'activity', 'datetime'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['duration'] = data.groupby(['location', 'activity','position'])['datetime'].diff().fillna(pd.Timedelta(seconds=0))
    data['duration_seconds'] = data['duration'].dt.total_seconds()
    data['total_duration_seconds'] = data.groupby(['date', 'location', 'position'])['duration_seconds'].transform('sum')
    picking_placing_data = data[data['activity'].isin(['picked', 'placed'])]

    # Group by date and activity, then count occurrences
    activity_counts = picking_placing_data.groupby(['date', 'activity']).size().reset_index(name='count')
    activity_counts=pd.DataFrame(activity_counts)
    return data,activity_counts

app = Flask(__name__)


@app.route('/')
def index():
    app_name = "Task Results"
    description = "This is a Flask web application for displaying task results."
    return render_template('index.html', app_name=app_name, description=description)


@app.route('/results', methods=['POST'])
def results():
    # Assuming data, data2, and data3 are accessible here
    data3 = pd.read_excel('rawdata.xlsx')
    data2 = pd.read_excel('test.xlsx')
    data = pd.read_excel('train.xlsx')
    clusters, score = task1(data, data2)
    training_score, y_pred_labels = task2(data, data2)
    df_duration, df_pick_and_place = task3(data3)

    # Pass the results to the results template
    return render_template('results.html',
                           clusters=clusters,
                           score=score,
                           training_score=training_score,
                           y_pred_labels=y_pred_labels,
                           df_duration=df_duration,
                           df_pick_and_place=df_pick_and_place)

if __name__ == '__main__':
    app.run(debug=True)


