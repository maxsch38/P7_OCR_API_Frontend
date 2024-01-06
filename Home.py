######################################################################################################################################################################
### Fichier Interface utilisateur Streamlit - Home
######################################################################################################################################################################

######################################################################################################################################################################
### Importation des librairies : 
import streamlit as st
from PIL import Image

######################################################################################################################################################################
### Données : 

# Définition de l'url de l'API - FastAPI : 
url = "https://apibackendp7-26621ca046b4.herokuapp.com"
st.session_state['url_api'] = url

# Initialisation de l'id_client : 
if 'id_client' not in st.session_state:
    st.session_state['id_client'] = None

# Initialisation de la prediction : 
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
    
# Importation du chemin du dossier de données : 
dossier_donnees = 'Data'

# Chargement du logo : 
logo = Image.open(dossier_donnees + '/Logo.png')

######################################################################################################################################################################
### Configuration de la page  : 
st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="auto",
    )

### Configuration spécifique pour handicap : 
specific_text_style = """
    <style>
        .specific-text {
            font-family: Verdana, sans-serif;
            font-size: 14px;
            font-weight: bold;
        }
    </style>
"""
######################################################################################################################################################################
### Interface : 

# Sidebar : 
grossisement = st.sidebar.toggle('Grossissement du texte')

# Application texte spécifique :
if grossisement:
    st.markdown(specific_text_style, unsafe_allow_html=True)
    
# Titre Intrface : 
st.title("Bienvenue sur l'API Prêt à Dépenser")

# Affichage du Logo : 
st.image(logo, width=200)

# Présentation : 
st.header("Présentation")

st.markdown(
    """
    <div class='specific-text'>

    Bienvenue sur l'API Prêt à Dépenser, un outil intelligent de scoring de crédit.  
    Il a pour mission d'apporter transparence et expertise à la gestion des demandes de crédit,  
    en utilisant des modèles de machine learning pour évaluer la probabilité de remboursement de chaque client.
    </div>
    """,
    unsafe_allow_html=True,
)

# Fonctionnalités : 
st.header("Fonctionnalités")
st.markdown(
    """
    <div class='specific-text'>
    
    ***1. Evaluation du dossier :***  
    Obtenez instantanément lévaluation du dossier (crédit accordé ou refusé) en fonction du risque de non-remboursement des échéances.  
    (*Le modèle de scoring analyse diverses sources de données pour fournir une prédiction précise.*)  
    
    Chaque évaluation est accompagnée d'un pourcentage de certitude, indiquant la confiance du modèle dans sa propre prédiction.

    ***2.Interprétation de l'évaluation :***  
    Comparez le dossier courant (dossier de l'évalutation) avec d'autres dossiers pour une compréhension appronfondie des résultats du modèle de scoring.  
    - Utilisez des tableaux permettant la comparaison des caractéristiques du dossier courant à des dossiers de références. 
    - Accédez à des visualisations graphiques (impact local et global des caractéristiques, comparaison avec des groupes de dossiers...)
    
    Cette interface offre une compréhension approfondie des résultats du modèle de scoring, permettant à l'utilisateur d'analyser les facteurs influants sur la prise de décision d'accord ou de refus du crédit.
    </div>
""",
unsafe_allow_html=True,
)