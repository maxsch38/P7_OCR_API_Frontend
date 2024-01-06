######################################################################################################################################################################
### Fichier Interface utilisateur Streamlit -  Evaluation
######################################################################################################################################################################

######################################################################################################################################################################
### Importation des librairies : 
import streamlit as st
import fct_interface

######################################################################################################################################################################
### Configuration de la page  : 

st.set_page_config(
    page_title="Evaluation du dossier",
    layout="wide",
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
## Récupération des données : 

# id_client : 
id_client = st.session_state.id_client

# prediction : 
prediction = st.session_state.prediction

# Récupération du threshold : 
threshold = fct_interface.recuperation_seuil()

######################################################################################################################################################################
### Interface : 

grossisement = st.sidebar.toggle('Grossissement du texte')

# Application texte spécifique :
if grossisement:
    st.markdown(specific_text_style, unsafe_allow_html=True)

# Titre de la page :
st.title("Evaluation du dossier", )
st.divider()

# Création de la sidebar :
st.sidebar.header("Evaluation du dossier")

# Input pour id_client
id_client = st.sidebar.number_input(
    label="Entrez le numéro du dossier client:",
    min_value=0,
    placeholder="N°dossier....",
    value=None,
    )

# Bouton de prédiction
bouton_prediction = st.sidebar.button(label='Evaluation')

# Action du bouton prédiction :
if bouton_prediction:
    if id_client:
    
        # Mise à jour de l'id_client : 
        st.session_state['id_client'] = id_client
        
        # Récupération du résultat de la prédiction :
        result = fct_interface.bouton_prediction(id_client=id_client)

        # Si le client est  présent dans la base de données :
        if result is not None:
            
            # Mise à jour de la prediction : 
            st.session_state['prediction'] = result['prediction']
            
            # Tracé et affichage des graphes de prédiction :
            figure_1 = fct_interface.graph_gauge(result, threshold)
            
            # Affichage : 
            st.plotly_chart(figure_1, use_container_width=True, width=400, height=50)
            st.divider()
            st.write(
                """
                <div class='specific-text'>

                **Note :**  
                Le score renvoyé est la probabilité que le crédit ne soit pas remboursé.  
                Seuil de décision optimum de {:.2f}.
                - Le crédit est considéré comme accordé pour une probabilté en dessous de ce seuil.
                - Le crédit est considéré comme refusé pour une probabilté au dessus de ce seuil. 
                *(Risque de non remboursement des échéances)*
                
                Il est également renvoyé le pourcentage de fiabilité de cette prédiction pour permettre une prise de décision plus éclairée.  
                Cette mesure rapporte la certitude d'une prédiction en évaluant la proximité de la probabilité prédite par rapport au seuil de décision.
                </div>
                """.format(threshold),
                unsafe_allow_html=True,
                )
            st.divider()
        
        elif result is None : 
            
            # Remise à None de la prediction 
            st.session_state['prediction'] = None
    else: 
        st.warning("Veuillez entrer un numéro de dossier client.")
        