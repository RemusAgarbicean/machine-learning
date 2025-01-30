import re
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

st.title('Detectarea fraudei in emailuri')

st.sidebar.title('Navigare')
page = st.sidebar.radio('Selecteaza pagina', ['Prezentare Modele', 'Aplicatie'])
models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
@st.cache_resource
def get_vectorizers():
    tfidf_vectorizer_subject = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer_body = TfidfVectorizer(max_features=1000)
    return tfidf_vectorizer_subject, tfidf_vectorizer_body


@st.cache_resource
def load_data():
        df = pd.read_csv('CEAS_08.csv')
        df.dropna(subset=['receiver'], inplace=True)
        df['subject'].fillna('No Subject', inplace=True)
        return df

@st.cache_resource
def create_features(df):
    tfidf_vectorizer_subject, tfidf_vectorizer_body = get_vectorizers()
    
    tfidf_subject = tfidf_vectorizer_subject.fit_transform(df['subject'])
    tfidf_body = tfidf_vectorizer_body.fit_transform(df['body'])

    # Add unique prefixes to avoid duplicate column names
    subject_df = pd.DataFrame(
        tfidf_subject.toarray(),
        columns=[f"subject_{name}" for name in tfidf_vectorizer_subject.get_feature_names_out()]
    )
    
    body_df = pd.DataFrame(
        tfidf_body.toarray(),
        columns=[f"body_{name}" for name in tfidf_vectorizer_body.get_feature_names_out()]
    )

    features = pd.concat([
        df[['urls']].reset_index(drop=True),
        subject_df,
        body_df
    ], axis=1)
    
    st.session_state['feature_names'] = features.columns.tolist()
    return features


@st.cache_resource
def train_models(features, df, _models):
    X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.2, random_state=42)
    results = []
    trained_models = {}  # NEW: Store fitted models
    
    for name, model in _models.items():
        model.fit(X_train, y_train)  # Train model
        trained_models[name] = model  # NEW: Store fitted model
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Accuracy Score': accuracy_score(y_test, y_pred)
        })
    
    return results, trained_models

df = load_data()
features = create_features(df)
results, trained_models = train_models(features, df, models)

if page == 'Prezentare Modele':
    st.header('Prezentare Modele')
    st.write('In aceasta aplicatie, vom folosi un set de date care contine informatii despre emailuri si daca acestea sunt sau nu frauduloase. Scopul nostru este sa antrenam un model care sa poata detecta daca un email este fraudulos sau nu.')

    st.write('''
             Acest dataset a fost compilat de cercetători pentru a studia tacticile emailurilor de phishing. 
                Combină emailuri dintr-o varietate de surse pentru a crea o resursă cuprinzătoare pentru analiză.
             ''')

    st.write('Începem prin a realiza importurile necesare pentru a începe analiza datelor.')
    st.code(''' 
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import TfidfVectorizer
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
''', language='python')

    st.write('''
                Încărcăm setul de date și afișăm primele câteva rânduri pentru a vedea cum arată.
                ''')

    st.code('''
            display_df = pd.read_csv('CEAS_08.csv')
            display_df.head()
            ''', language='python')
    display_df = pd.read_csv('CEAS_08.csv')
    st.write(display_df.head())

    st.write('''
                Rulam isnull().sum() pentru a verifica daca exista valori lipsa in setul de date.
                ''')

    st.code('''
            display_df.isnull().sum()
            ''', language='python')

    st.write(display_df.isnull().sum())

    st.write('''
                Eliminam randurile care nu au receiver si inlocuim valorile lipsa din subject cu No Subject.
                ''')

    st.code('''
            display_df.dropna(subset=['receiver'], inplace=True)
            display_df['subject'].fillna('No Subject', inplace=True)
            ''', language='python')
    display_df.dropna(subset=['receiver'], inplace=True)
    display_df['subject'].fillna('No Subject', inplace=True)
    st.write('''
                Verificam daca mai exista valori lipsa.
                ''')

    st.code('''
            display_df.isnull().sum()
            ''', language='python')

    st.write(display_df.isnull().sum())

    st.write('''
                Vizualizam distributia label si url in setul de date.
                Acestea sunt variabile categorice, deci vom folosi countplot.
                Restul variabilelor sunt, in mare parte, de tip string
                ''')

    st.code('''
            fig, ax = plt.subplots()
    sns.countplot(x='label', data=display_df, ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.countplot(x='urls', data=display_df, ax=ax)
    st.pyplot(fig)
            ''', language='python')
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=display_df, ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.countplot(x='urls', data=display_df, ax=ax)
    st.pyplot(fig)

    st.write('''
            Transformăm textul din coloanele subject și body în reprezentări numerice folosind TF-IDF 
            (Term Frequency-Inverse Document Frequency). Aceasta este o tehnică de extragere a caracteristicilor
            care reflectă importanța unui termen în document, ținând cont de frecvența acestuia în toate documentele.
            Limitează numărul de caracteristici la 1000 pentru a reduce dimensiunea spațiului de caracteristici și a 
            îmbunătăți eficiența modelului de machine learning
                ''')



    st.code('''
        @st.cache_resource
        def get_vectorizers():
        tfidf_vectorizer_subject = TfidfVectorizer(max_features=1000)
        tfidf_vectorizer_body = TfidfVectorizer(max_features=1000)
        return tfidf_vectorizer_subject, tfidf_vectorizer_body
            ''', language='python')


    st.write('''
            Combinăm caracteristicile extrase din coloanele urls, subject și body într-un singur DataFrame features. Acest DataFrame va fi folosit ulterior ca set de caracteristici pentru antrenarea modelului.
            ''')

    st.code('''
        @st.cache_resource
        def create_features(df):
        tfidf_vectorizer_subject, tfidf_vectorizer_body = get_vectorizers()
        
        tfidf_subject = tfidf_vectorizer_subject.fit_transform(df['subject'])
        tfidf_body = tfidf_vectorizer_body.fit_transform(df['body'])

        # Add unique prefixes to avoid duplicate column names
        subject_df = pd.DataFrame(
                tfidf_subject.toarray(),
                columns=[f"subject_{name}" for name in tfidf_vectorizer_subject.get_feature_names_out()]
        )
        
        body_df = pd.DataFrame(
                tfidf_body.toarray(),
                columns=[f"body_{name}" for name in tfidf_vectorizer_body.get_feature_names_out()]
        )

        features = pd.concat([
                df[['urls']].reset_index(drop=True),
                subject_df,
                body_df
        ], axis=1)
        
        st.session_state['feature_names'] = features.columns.tolist()
        return features
            ''', language='python')


    st.write('''
            Împărțim setul de date în set de antrenare și set de testare folosind train_test_split.
            Vom folosi 80% din date pentru antrenare și 20% pentru testare.
            ''')


    st.code('''
            X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.2, random_state=42)
            ''', language='python')

    st.write('''
            Vom folosi mai multe modele de clasificare pe care le vom testa si evalua la final pentru a vedea care este cel mai accurate.
             ''')

    st.code('''
        models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        @st.cache_resource
        @st.cache_resource
        def train_models(features, df, _models):
        X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.2, random_state=42)
        results = []
        trained_models = {}  # NEW: Store fitted models
        
        for name, model in _models.items():
                model.fit(X_train, y_train)  # Train model
                trained_models[name] = model  # NEW: Store fitted model
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                results.append({
                'Model': name,
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'Accuracy Score': accuracy_score(y_test, y_pred)
                })
    
            ''', language='python')


    for result in results:
        st.write(f"Model: {result['Model']}")
        st.write(f"Confusion Matrix: {result['Confusion Matrix']}")
        st.write(f"Accuracy Score: {result['Accuracy Score']}")
        st.write("="*60)

    st.write('''
            Pentru că modelul Vector Support Machine a obținut cel mai bun scor de acuratețe, vom folosi acest model pentru a face predicții în aplicația noastra.
            ''')

elif page == 'Aplicatie':
    st.header('Aplicatie')
    st.write('''
            În această aplicație, vom folosi modelul Support Vector Machine antrenat anterior pentru a face predicții pe datele introduse de utilizator.
            ''')

    st.write('''
            Introduceti subiectul și conținutul emailului pentru a determina dacă este fraudulos sau nu.
            ATENȚIE: modelul a fost antrenat pe emailuri în limba engleză, deci rezultatele pot fi irelevante pentru emailuri în alte limbi.
            ''')
    tfidf_vectorizer_subject, tfidf_vectorizer_body = get_vectorizers()
    subject = st.text_input('Subject')
    body = st.text_area('Body')

    def check_urls(text):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        return 1 if len(urls) > 0 else 0

    

    if st.button('Predict'):
        urls = check_urls(body)
        
        # Transform with prefixes
        subject_features = tfidf_vectorizer_subject.transform([subject])
        body_features = tfidf_vectorizer_body.transform([body])
        
        # Create DataFrames with matching prefixed columns
        subject_df = pd.DataFrame(
                subject_features.toarray(),
                columns=[f"subject_{name}" for name in tfidf_vectorizer_subject.get_feature_names_out()]
        )
        
        body_df = pd.DataFrame(
                body_features.toarray(),
                columns=[f"body_{name}" for name in tfidf_vectorizer_body.get_feature_names_out()]
        )

        X = pd.concat([
                pd.DataFrame({'urls': [urls]}),
                subject_df,
                body_df
        ], axis=1)
        
        # Align columns (no duplicates now)
        X = X.reindex(columns=st.session_state['feature_names'], fill_value=0)
        
        prediction = trained_models["Support Vector Machine"].predict(X)  # Use stored model
        
        if prediction[0] == 0:
            st.error('Există o șansă mare ca emailul să fie fraudulos!')
        else:
            st.success('Șansele ca emailul să fie fraudulos sunt mici.')