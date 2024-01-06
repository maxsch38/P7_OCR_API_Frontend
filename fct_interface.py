# Fichier des fonctions pour l'interface utilisateur.

######################################################################################################################################################################
### Importation des librairies : 
import pandas as pd
import numpy as np
import pickle
import gzip
import requests
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import euclidean_distances
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns
import shap
from io import BytesIO

###########################################################################################################################
#### Fonction de gestion des données pickle et gzip : 

def chargement_pickle(name, chemin): 
    path = chemin + '/' + name + '.pickle'

    with open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier


def enregistrement_pickle(name, chemin, fichier):    
    path = chemin + '/' + name + '.pickle'

    with open(path, 'wb') as f:
        pickle.dump(fichier, f)
        
def chargement_pickle_gzip(name, chemin):
    path = chemin + '/' + name + '.pickle.gz'

    with gzip.open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier


def enregistrement_pickle_gzip(name, chemin, fichier):
    path = chemin + '/' + name + '.pickle.gz'

    with gzip.open(path, 'wb') as f:
        pickle.dump(fichier, f)
 
        
######################################################################################################################################################################
### Charement des données nécessaire : 
url = st.session_state.url_api

# Chemin du dossier sauvegarde : 
dossier_sauvegarde = 'Data'

# Data test : 
data_test = chargement_pickle_gzip(
    name='data_final_test',
    chemin=dossier_sauvegarde,
)

# Data train : 
data_train = chargement_pickle_gzip(
    name='data_final_train',
    chemin=dossier_sauvegarde,
    )

######################################################################################################################################################################
### Fonctions  : 

def recuperation_seuil(): 
    """
    Fonction pour récupérer le seuil spécifique depuis l'API externe.

    Returns:
        float or None: La valeur du seuil spécifique si la requête est réussie, sinon None.
    """
    
    # Effectuer une requête GET à l'API
    reponse = requests.get(url+"/get_threshold")

    # Vérification de la requête:
    if reponse.status_code == 200:
        result  = reponse.json()
        return result['Threshold']
        
    else:
        st.error(f"Erreur lors du chargement du seuil spécifique: {reponse.status_code}")
        return None


###############################################################
@st.cache_data
def bouton_prediction(id_client):
    """
    Effectue une prédiction pour un client spécifié en utilisant l'API externe.

    Args:
        id_client (int): Identifiant du client pour lequel la prédiction doit être effectuée.

    Returns:
        dict or None: Un dictionnaire contenant les résultats de la prédiction, ou None si une erreur s'est produite.
    """

    # Filtre sur data_test : 
    X = data_test[data_test.index == id_client]
    
    # Test de la présence de l'id_client dans data_test : 
    if X.empty:
        st.error(f"Erreur 404. Le dossier client N°{id_client} n'existe pas dans la base de données.")
        return None
    
    # Envoie de la requête à l'API : 
    payload = {"X": X.to_dict(orient="records")[0]}
    response = requests.post(url+"/predict", json=payload)
      
    # Vérification du status_code de la réponse : 
    if response.status_code == 200:
        try:
            result = response.json()
            return result
       
        except Exception as e:
            st.error(f"Erreur lors de la conversion JSON : {e}")
    
    else:
        st.error(f"La requête à l'API a échoué, Erreur : {response.status_code}")
        
###############################################################
def graph_gauge(result, threshold):
    """
     Crée et retourne une figure Plotly représentant les résultats d'une prédiction sous forme de jauges.

    Args:
        result (dict): Dictionnaire contenant les résultats de la prédiction, généralement obtenu à partir de la fonction bouton_prediction.
            - 'prediction' (str): La prédiction résultante.
            - 'certitude' (float): Le niveau de certitude associé à la prédiction (entre 0 et 1).
            - 'proba' (float): La probabilité associée à la prédiction (entre 0 et 1).
        threshold (float): Seuil de décision spécifique.

    Returns:
        go.Figure: Figure Plotly représentant les jauges de probabilité et de certitude ainsi que l'affichage de la prédiction.
    """
    # Récupération des données : 
    prediction = result['prediction']
    certitude = result['certitude']
    proba = result['proba']
    
    # Création de la figure :
    fig_1 = go.Figure()
    
    # Ajout du titre de la figure
    fig_1.update_layout(
            title={'text': 'Résultat',
                   'font': {'size': 30},
                   'y':0.9,
                   'x':0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            )
    
    
    # Création de la jauge de probabilité :
    fig_1.add_trace(go.Indicator(
    mode="gauge+number",
    value=proba,
    title={'text': "Probabilté", 'font': {'size': 16}},
    gauge={
        'shape': "bullet",
        'axis': {
            'range': [0, 1],
        },
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 1,
            'value': threshold,
        },
        'steps': [
            {'range': [0, threshold], 'color': "green"},
            {'range': [threshold, 1], 'color': "crimson"}
        ],
        'bar': {
            'color': "grey",
            'thickness': 0.2,
        },
    },
    domain={'x': [0.25, 0.75], 'y': [0.85, 1]},
    ))
      
    # Création de la jauge de certitude : 
    fig_1.add_trace(go.Indicator(
        mode="gauge+number",
        value=certitude*100,
        number = {"suffix": "%"},
        title={'text': "Jauge de certitude", 'font': {'size': 16}},
        gauge={
            'axis': {
            'range': [0, 100],
        },
            'steps': [
                {'range': [0, 100/3], 'color': "crimson"},
                {'range': [100/3, 200/3], 'color': "darkorange"},
                {'range': [200/3, 100], 'color': 'green'}
                ],
            'bar': {
                'color': "grey",
                'thickness': 0.2,
                },
            },
        domain={'x': [0.05, 0.45], 'y': [0, 0.45]},
        ))
    
    # Création l'affichage de la prédiction :
    fig_1.add_annotation(
    text=f"Crédit\n {prediction}",
    x=0.8,  # Position x du texte
    y=0.2,  # Position y du texte
    showarrow=False,
    font={
        'family': "Arial",
        'size': 30,
        'color': 'green' if prediction == 'Accordé' else 'crimson',
    }
)
    
    return fig_1

###############################################################
@st.cache_data
def recuperation_index_ref():
    return data_train.index

###############################################################
@st.cache_data
def recuperation_variables():
    ls = data_train.columns.tolist()
    ls.remove('TARGET')
    ls = sorted(ls)
    return ls
    
###############################################################
@st.cache_data
def creation_df_choix_perso(id_client, id_ref):
    """
    Crée un DataFrame combinant les données du dossier courant et du dossier de référence.

    Args:
        id_client (type): L'identifiant du dossier courant.
        id_ref (type): L'identifiant du dossier de référence.

    Returns:
        pd.DataFrame: Un DataFrame combinant les données du dossier courant et du dossier de référence.
    """
    
    # Filtre du data_test: 
    data_courant = data_test[data_test.index == id_client]
    
    # Filtre du data_train : 
    data_ref = data_train[data_train.index == id_ref]
    data_ref = data_ref.drop('TARGET', axis=1)
    
    # Transposition et renommage des colonnes : 
    data_courant = data_courant.transpose()
    data_courant = data_courant.rename(columns={id_client: 'Dossier Courant'})
    
    data_ref = data_ref.transpose()
    data_ref = data_ref.rename(columns={id_ref: 'Dossier de Référence'})
    
    # Merge des deux dataframes en un seul : 
    df = data_courant.merge(data_ref, left_index=True, right_index=True)
    
    # Trie par ordre alphabétiques des variables : 
    df = df.sort_index()
    
    return df


###############################################################
@st.cache_data
def get_statut_ref(id_ref):
    """
    Récupère le statut du dossier de référence. (Réelle atttribution ou non du crédit)

    Args:
        id_ref (type): Numéro du dossier client de référence.

    Returns:
        str: Le statut de la référence, qui peut être 'Accordé' ou 'Refusé'.
    """
    
    # Filtre sur data_train : 
    data_ref = data_train[data_train.index == id_ref]
    
    # Récupération de la valeur de la TARGET : 
    target = data_ref['TARGET'].values[0]
    
    # Création du statut : 
    statut = 'Accordé' if target == 0 else 'Refusé'

    return statut

###############################################################
@st.cache_data
def graph_fetaures_importance(n_top=20):
    """
    Génère un graphique en barres visualisant l'importance des caractéristiques en fonction du modèle de l'API.

    Args:
        n_top (int, optional): Le nombre de caractéristiques à afficher dans le graphique. Par défaut, 20.

    Returns:
        matplotlib.figure.Figure: Une figure Matplotlib représentant l'importance des caractéristiques.
    """

    
    # Effectuer une requête GET à l'API
    reponse = requests.get(url+"/get_features_importance")

    # Vérification de la requête:
    if reponse.status_code == 200:
        result  = reponse.json()
         
    else:
        st.error(f"Erreur lors du chargement des features importance: {reponse.status_code}")
        return None
    
    # Création d'une liste de features_importance normalisée entre 0 et 100 : 
    importance = result['Importance']
    min_val = min(importance)
    max_val = max(importance)

    importance = [(val - min_val) / (max_val - min_val) * 100 for val in importance]

    # Création d'un DataFrame à partir de la liste de features importance : 
    df= pd.DataFrame({
        'Feature': [col for col in data_test.columns.to_list()],
        'Importance': importance,
        })

    # Trie de df : 
    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    df = df.head(n_top)

    # Création du graphique : 
    fig_height = max(3, n_top * 0.3)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    sns.barplot(
        x="Importance",
        y='Feature',
        data=df,
        ax=ax
    )

    plt.tight_layout()
    
    return fig
    
###############################################################
@st.cache_data
def creation_df_choix_auto(id_client):
    """
    Crée un DataFrame comparant les caractéristiques du dossier courant avec les trois dossiers de référence les plus proches.

    Args:
        id_client (int): L'identifiant du dossier client.

    Returns:
        pd.DataFrame: Un DataFrame contenant les caractéristiques du dossier courant et des trois dossiers de référence les plus proches.
        list: Une liste des identifiants des dossiers de référence sélectionnés.
    """

    # Filtre du data_test: 
    data_courant = data_test[data_test.index == id_client]
    
    # Filtre du data_train sur les crédits acccordés :
    data_ref = data_train.loc[data_train['TARGET'] == 0]
    data_ref = data_ref.drop('TARGET', axis=1)
    
    # Normalisation des données :
    scaler = MinMaxScaler()
    data_ref_normalized = pd.DataFrame(scaler.fit_transform(data_ref), index= data_ref.index, columns=data_ref.columns)
    data_courant_normalized = pd.DataFrame(scaler.transform(data_courant), index=data_courant.index, columns=data_courant.columns)

    # Calcul des distances euclidiennes entre le dossier courant et les dossiers de data_train :
    distances = euclidean_distances(data_courant_normalized, data_ref_normalized)
  
    # Sélection des 3 dossiers les plus proches du dossier courant : 
    indices = distances.argsort()[0][:3]
    ref_index = data_ref.index[indices]
    
    data_ref = data_ref[data_ref.index.isin(ref_index)].reindex(ref_index)
    
    # Transposition et renommage des colonnes : 
    data_courant = data_courant.transpose()
    data_courant = data_courant.rename(columns={id_client: 'Dossier Courant'})
    
    data_ref = data_ref.transpose()
    data_ref = data_ref.rename(columns={
        ref_index[0]: 'Dossier de Référence 1',
        ref_index[1]: 'Dossier de Référence 2',
        ref_index[2]: 'Dossier de Référence 3',
        })
    
    # Merge des deux dataframes en un seul : 
    df = data_courant.merge(data_ref, left_index=True, right_index=True)
    
    # Trie par ordre alphabétiques des variables : 
    df = df.sort_index()
    
    return df, ref_index

###############################################################
@st.cache_data
def shap_waterfall_plot(id_client):
    """
    Génère un graphique waterfall_plot illustrant l'impact des caractéristiques sur la prédiction.

    Args:
        id_client (type): Numéro de dossier client.

    Returns:
        matplotlib.figure.Figure: Une figure Matplotlib affichant le graphique waterfall_plot.
    """
    
    # Filtre sur data_test : 
    X = data_test[data_test.index == id_client]
    
    # Envoie de la requête à l'API : 
    payload = {"X": X.to_dict(orient="records")[0]}
    response = requests.post(url+"/get_shap_values", json=payload)
    
    # Vérification du status_code de la réponse : 
    if response.status_code == 200:
        try:
            result = response.json()
            shap_values = np.array(result['shap_values'])
            base_values = result['base_values']
            data_values = np.array(result['data_values'])
            
        except Exception as e:
            st.error(f"Erreur lors de la conversion JSON : {e}")
    
    else:
        st.error(f"Erreur lors du chargement des shap_values : {response.status_code}")
       
    # Création d'un objet Explanation avec les noms de caractéristiques : 
    exp = shap.Explanation(
        values=shap_values, 
        base_values=base_values, 
        data=data_values, 
        feature_names=data_test.columns,
        )
    
    # Création de la figure waterfall_plot : 
    fig, ax = plt.subplots()
    
    shap.waterfall_plot(exp, max_display=11)
    
    return fig

###############################################################
@st.cache_data
def diagramme_radar(id_client, ls_var, comp='tous'):
    """
    Génère un diagramme en radar comparant les caractéristiques du dossier courant avec la moyenne des caractéristiques
    pour l'ensemble des dossiers acceptés, l'ensemble des dossiers refusés, ou l'ensemble des dossiers de la base de données.

    Args:
        id_client (int): L'identifiant du dossier client.
        ls_var (List[str]): La liste des variables à inclure dans le diagramme en radar.
        comp (str, optional): Le groupe de dossiers à comparer ('tous', 'acceptés', 'refusés'). Par défaut, 'tous'.

    Returns:
        BytesIO: Un objet BytesIO contenant l'image du diagramme en radar.
    """
    
    # Filtre su data_test : 
    df_courant = data_test.loc[data_test.index == id_client, ls_var]
    
    # Filtre sur data_train : 
    df = data_train[ls_var + ['TARGET']]
    
    # Normalisation Min-Max pour chaque variable par rapport à elle-même dans data_train : 
    df_normalized = (df[ls_var] - data_train[ls_var].min()) / (data_train[ls_var].max() - data_train[ls_var].min())
    
    # Normalisation du dossier courant par rapport à data_train : 
    df_courant_normalized = (df_courant - data_train[ls_var].min()) / (data_train[ls_var].max() - data_train[ls_var].min())

    # Calcul de la moyenne des variables normalisées pour chaque groupe : 
    if comp == 'tous':
        mean_values = df_normalized.mean()
        titre = "Moyenne sur l'ensemble des dossiers acceptés/refusés"
    elif comp == 'acceptés': 
        mean_values= df_normalized[df['TARGET'] == 0].mean()
        titre = "Moyenne sur l'ensemble des dossiers acceptés"

    else: 
        mean_values = df_normalized[df['TARGET'] == 1].mean()
        titre = "Moyenne sur l'ensemble des dossiers refusés"

    # Création du diagramme en radar : 
    categories = ls_var
    N = len(categories)

    # Calcul des angles des axes pour chaque caractéristique : 
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Fermeture de la figure : 
    plt.close()  
    
    # Création d'une figure avec des sous-plots polarisés : 
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Ajoutez des axes : 
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Dessin des lignes pour chaque caractéristique : 
    ax.plot(angles, mean_values.values, 'o-', linewidth=2, label=titre)
    ax.fill(angles, mean_values.values, alpha=0.25)

    # Ajout des valeurs du dossier courant : 
    ax.plot(angles, df_courant_normalized.values.flatten(), 'o-', linewidth=2, label='Dossier courant')
    ax.fill(angles, df_courant_normalized.values.flatten(), alpha=0.25)

    # Positionne la légende à l'extérieur du cercle
    ax.legend(loc='upper right', bbox_to_anchor=(1.7, 1), fontsize=8)  
    
    # Ajuste l'espacement entre les catégories pour éviter le chevauchement
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=8)
    
     # Enregistrer la figure dans un buffer BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    
    # Afficher l'image avec une largeur spécifiée
    
    return buf
