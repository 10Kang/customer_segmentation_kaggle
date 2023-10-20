# Import library 
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from scipy.stats.mstats import trimmed_var
from dash import html, dcc, Output,Input
from dash import Dash
from dash import dash_table
from dash import callback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Read the data 
df = pd.read_csv('/Users/kang/Documents/Portfolio/Customer-data/marketing_campaign.csv',delimiter='\t')

# Drop the Null rows
df= df.dropna()

# Drop categorical data
df_all_features = df.drop(columns=['ID','Year_Birth', 'Education', 'Marital_Status','Dt_Customer','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'])

# Instantiate a trimmed variance table to be displayed 
top_five_features = (
            df_all_features.apply(trimmed_var,limits=(0.1,0.1)).sort_values().tail(6)
        )
        
# putting in value to new dataframe 
data = top_five_features.to_frame().reset_index().rename(columns={'index':'Features',0:'Variances'})


# instantiate app
app = Dash(__name__)

app.layout = html.Div(
    # style={'backgroundColor': 'black'},
    # children=
    [
        # Application title
        html.H1("Customer Segmentation",
                style={'textAlign': 'center'}
               ),
        # bar chart element 
        html.H2("High Variance Features",
                style={'textAlign': 'center'}
               ),
        
        # radio component
        dcc.RadioItems(
            options=[
                {'label':"Trimmed",'value':True},
                {'label':'Not Trimmed','value':False}
            ]
            ,value=True,
            id='trim-button'),
        
        # table component 
         dash_table.DataTable(
             id='table',
             columns=[{'name': col, 'id': col} for col in data.columns],
             data=data.to_dict('records')
         ),
        
        # 2nd block
        html.H2('K-mean clustering',
                style={'textAlign': 'center'}
               ),
        html.H3('Number of cluster (k)',
                style={'textAlign': 'center'}
               ),
        
        # slider 
        dcc.Slider(id='k-slider',min=2,max=10,step=1,value=2),
        html.Div(id='metrics'),
        
        #3rd block
        #PCA plot
        dcc.Graph(id='PCA-plot')
        
        
    ]
)

def get_high_var_features(trimmed=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    # # acess global variances
    # global top_five_features
    # calculate the variance
    
    if trimmed==True:
        top_five_features = (
            df_all_features.apply(trimmed_var,limits=(0.1,0.1)).sort_values().tail(6)
        )
        
        # putting in value to new dataframe 
        top_five_features = top_five_features.to_frame().reset_index().rename(columns={'index':'Features',0:'Variances'})
        return top_five_features#to_dict('records')
    else:
        top_five_features_no_trimmed = df_all_features.var().sort_values().tail(6)
        
        # putting in value to new dataframe
        top_five_features_no_trimmed = top_five_features_no_trimmed.to_frame().reset_index().rename(columns={'index':'Features',0:'Variances'})

        return top_five_features_no_trimmed#
    

def get_model_metrics(trimmed=True,return_metrics=False,k=2):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    
    # get the features
    
    features = get_high_var_features(trimmed=trimmed)
    
    # create feature matrix
    selected_features = features['Features'].values

    df_features = df_all_features[selected_features]
    
    # make model
    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=k,random_state=10)
    )
    # fit model
    model.fit(df_features)
    
    if return_metrics == True:
        # calculate inertia
        inertia = model.named_steps['kmeans'].inertia_
        
        #calculate silhoutte score
        ss = silhouette_score(df_features,model.named_steps['kmeans'].labels_)
        
        # put into the dictionary
        metrics = {
            "Inertia":round(inertia),
            "Silhouette": round(ss,3)
        }
        
        return metrics
    
    return model

def get_pca_labels(trimmed=True,k=2):

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    
    # get the features
    features = get_high_var_features(trimmed=trimmed)
    
    # create feature matrix
    selected_features = features['Features'].values
    df_features = df_all_features[selected_features]
    
    # PCA transformer
    pca = PCA(n_components=2,random_state=10)
    
    # transform data
    X_t = pca.fit_transform(df_features)
    
    # put into dataframe
    
    X_pca = pd.DataFrame(X_t,columns=['PC1','PC2'])
    
    # add labels 
    
    model = get_model_metrics(trimmed=trimmed,return_metrics=False,k=k)
    
    X_pca['labels'] = model.named_steps['kmeans'].labels_.astype(str)
    X_pca.sort_values("labels",inplace=True)
    return X_pca

@app.callback(
    Output("table","data"),Input("trim-button","value")
)
def serve_table(trimmed=True):
    if trimmed==True:
        return get_high_var_features(trimmed=trimmed).to_dict('records')
    else:
        return get_high_var_features(trimmed=trimmed).to_dict('records')

    
@app.callback(
    Output('metrics','children'),
    Input('trim-button','value'),
    Input('k-slider','value')
)
def serve_metrics(trimmed=True,k=2):

    """Returns list of ``H3`` elements containing inertia and silhouette score
    for ``KMeans`` model.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    
    #get metrics
    metrics = get_model_metrics(trimmed=trimmed,return_metrics=True,k=k)
    
    # add metrics to HTML elements
    text = [
        html.H3(f"Inertia :{metrics['Inertia']}",style={'textAlign': 'center'}),
        html.H3(f"Silhouette Score :{metrics['Silhouette']}",style={'textAlign': 'center'})
    ]
    
    return text

@app.callback(
    Output('PCA-plot','figure'),
    Input('trim-button','value'),
    Input('k-slider','value')

)
def serve_scatter_plot(trimmed=True,k=2):

    """Build 2D scatter plot of ``df`` with ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    
    df = get_pca_labels(trimmed=trimmed,k=k)
    
    fig = px.scatter(data_frame=df,x='PC1',y='PC2',color='labels',
                    title='PCA Representation of clusters')
    
    fig.update_layout(xaxis_title='PC1',yaxis_title='PC2')
    
    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)