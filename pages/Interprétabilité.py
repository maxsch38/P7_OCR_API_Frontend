######################################################################################################################################################################
### Fichier Interface utilisateur Streamlit - Interprétabilité
######################################################################################################################################################################

######################################################################################################################################################################
### Importation des librairies : 
import streamlit as st
import fct_interface

######################################################################################################################################################################
### Configuration de la page  : 

st.set_page_config(
    page_title="Interpretation",
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
### Récupération des données : 

# id_client : 
id_client = st.session_state.id_client

# Prédiction du dossier courant : 
prediction_courant = st.session_state.prediction

######################################################################################################################################################################
### Interface : 
grossisement = st.sidebar.toggle('Grossissement du texte')

# Application texte spécifique :
if grossisement:
    st.markdown(specific_text_style, unsafe_allow_html=True)

# Titre de la page : 
st.title("Interprétabilité de l'évaluation")

with st.expander("Notice de la page"):
    st.write(
    """
    <div class='specific-text'>

    **<u>Barre latérale :</u>**
    - *Informations Dossier Courant :*
        - Dossier Courant: numéro dossier sélectionné dans la pages Evaluation.
        - Statut: Prédiction pour l'accord ou le refus du crédit pour se dossier.
    - *Paramètre du tableau de comparasion :*
        - mode 'Auto' :  Comparaison du dossier courant avec les 3 dossiers les plus proches possédant un crédit accordé.  
        - mode 'Perso' : Comparaison du dossier courant avec le dossier de votre choix.
    
    **<u>Page Principale :</u>**
    - *Tableau de comparaison :* Affichage du tableau en fonction des paramètres sélectionnés.
    - *Visualisations Graphiques :*
        - Impacts Spécifiques : Affiche les impacts spécifiques des 10 caractéristiques les plus importantes sur la prédiction.
        - Comparaison : Permet de comparer le dossier courant avec différents groupes de dossiers en fonction des paramètres choisis (variables, groupe de comparaison..).
        - Impacts Globaux: Met en évidence l'importance globale de chaque caractéristique sur les prédictions du modèle.
    </div>
    """,
    unsafe_allow_html=True,
    )

st.divider()

# Sidebar : 
st.sidebar.markdown("#### <u>Dossier courant :</u>", unsafe_allow_html=True)

#########################################################################################
# Affichage du dossier courant : 
if st.session_state.id_client is not None: 
    st.sidebar.write(f"N°dossier : {id_client}")
    
    # Affichage du statut du dossier courant : 
    if prediction_courant: 
        st.sidebar.write(f"Statut : {prediction_courant}")
    else: 
        st.sidebar.markdown(
            """
            Statut :  
            *Le dossier n'est pas dans la base de données.*
            """
            )
        
        st.warning("Interprétabilité impossible, dossier courant inconnu.")

else : 
     st.sidebar.caption("Aucun dossier courant sélectionné.")
     
st.sidebar.divider()
#########################################################################################
# Affichage de la selction du tableau de comparaison : 

# initialisation : 
auto = False
perso = False

if prediction_courant is not None: 
    choice = st.sidebar.radio(
        "#### Tableau de comparaison",
        ['Auto', 'Perso'],
        horizontal=True,
    )

    if choice == 'Auto': 
        auto = True
            
    else : 
        perso = True
       
#########################################################################################
# Choix de méthode Perso - Affichge du DataFrame de comparaison entre dossier courant et un dossier de référence choisit par l'utilisateur : 
if perso:
        
    # Affichage de la sidebar : 
    id_ref = st.sidebar.selectbox(
        label='Dossier de référence :',
        options=fct_interface.recuperation_index_ref(),
        placeholder='N°Dossier de réfrence...)',
        )
    
    statut_ref = fct_interface.get_statut_ref(id_ref=id_ref)
    st.sidebar.write(f"Statut : {statut_ref}")
        
        
    # Affichage du tableau de comparasion :  
    st.markdown(f"#### Tableau de comparaison entre le dossier courant N°{id_client} et le dossier référence N°{id_ref}")
                  
    df = fct_interface.creation_df_choix_perso(
        id_client=id_client, 
        id_ref=id_ref,
    )

    # Affichage du DataFrame : 
    surlignage = st.toggle("Surlignage de la valeur la plus élevée entre les deux dossiers.")

    if surlignage : 
        st.dataframe(
            df.style
            .highlight_max(axis=1,color='lightgreen')
            .format("{:.3f}"),
            use_container_width=True,
            )
    
    else: 
        st.dataframe(
            df.style.format("{:.3f}"),
            use_container_width=True,
            )
        
    st.sidebar.divider()
    st.divider()
        
#########################################################################################
# Choix de méthode Auto - Affichge du DataFrame de comparaison entre dossier courant et les 3 dossiers de référence les plus proche possédant un crédit accordé : 

if auto: 
        
    # Récupération de df et des numéros de dossiers de ref : 
    df, dossiers_ref = fct_interface.creation_df_choix_auto(
        id_client=id_client,
        )
        
    # Affichage de la sidebar : 
    st.sidebar.write(
        f"""
        Dossiers de références :
              
        - N°dossier_ref_1 ==> {dossiers_ref[0]}  
        - N°dossier_ref_2 ==> {dossiers_ref[1]}  
        - N°dossier_ref_3 ==> {dossiers_ref[2]}  
            
        Statut : Accordé
        """
        )

    # Affichage du tableau de comparasion :  
    st.markdown(f"#### Tableau de comparaison entre le dossier courant N°{id_client} et les dossiers de références.")
    
    st.dataframe(
        df.style.format("{:.3f}"),
        use_container_width=True,
        )
    
    st.sidebar.divider()
    st.divider()
      
#########################################################################################
# Visualisations graphiques : 
if prediction_courant is not None:
    
    # Affichage du titre : 
    st.markdown(f"#### Visualisations graphiques")
    
    # Création des tabs : 
    tab1, tab2, tab3 = st.tabs(["Impacts spécifiques", "Comparaison", "Impacts globaux"])
    
    with tab1:
        st.write(
            f"<div class='specific-text'>Impacts spécifiques des 10 caractéristiques les plus importantes pour la prédiction sur le dossier courant N°{id_client}</div>",
            unsafe_allow_html=True,
            ) 
        
        waterfall_fig = fct_interface.shap_waterfall_plot(id_client=id_client)
        st.pyplot(waterfall_fig)
        
    with tab2: 
        st.write(
            f"<div class='specific-text'>Comparaison du dossier courant N°{id_client} en fonction des paramètres choisis</div>",
            unsafe_allow_html=True,
            )
        st.caption(
            "<div class='specific-text'>Les valeurs sont normalisées entre 0 et 1 pour une meilleure représentation.</div>",
            unsafe_allow_html=True,
            )         
        
        mode = st.radio(
            "Avec quel groupe souhaitez vous comparer le dossier courant",
            ['Tous les dossiers', 'Tous les dossiers acceptés', 'Tous les dossiers refusés'],
            horizontal=True,
        )
        
        if mode == 'Tous les dossiers': 
            comp = 'tous'
        elif mode == 'Tous les dossiers acceptés':
            comp = 'acceptés'
        else:
            comp = 'refusés'
        
        liste_variables = st.multiselect(
            label='Variables à comparer :',
            options=fct_interface.recuperation_variables(),
            placeholder='Sélectionnez des options...',
        )
    
        if len(liste_variables) == 0 : 
            st.warning('Aucunes variables sélectionnées.')
        else: 
            fig = fct_interface.diagramme_radar(id_client, liste_variables, comp=comp)
            st.image(fig, width=800)
    
    with tab3:
        st.write(
            "<div class='specific-text'>Importance globale des caractéristiques, mise en évidence des influences générales de chaque caractéristique sur les prédictions du modèle.</div>",
            unsafe_allow_html=True,
            )        
        
        n_features_global = st.number_input(
            label='Entrez le nombre de caractéristiques que vous souhaitez afficher', 
            min_value=1,
            max_value=len(fct_interface.recuperation_variables()),
            placeholder='...',
            value=20,
            )
        
        features_importances_fig = fct_interface.graph_fetaures_importance(n_top=n_features_global)

        st.pyplot(features_importances_fig)
        
