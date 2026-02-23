"""
generate_report.py  -  Generate PDF Report of the Pipeline Explanation

Generates a professional, pedagogical PDF document covering all pipeline
steps, suitable for oral defense (soutenance), competition presentation,
and README documentation.

Author: Yomna Haouel
"""

from fpdf import FPDF
import os

class PipelineReport(FPDF):
    """Custom PDF class with header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Systeme de Prediction de Defaut de Paiement  -  Rapport Complet", align="C")
        self.ln(4)
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Yomna Haouel  -  Page {self.page_no()}/{{nb}}", align="C")

    # ── Helper methods ──

    def section_title(self, title, level=1):
        """Add a section title."""
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(41, 128, 185)
            self.ln(4)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(41, 128, 185)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(44, 62, 80)
            self.ln(3)
            self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(52, 73, 94)
            self.ln(2)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def body_text(self, text):
        """Add body text paragraph."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        """Add bold text."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic_text(self, text):
        """Add italic text."""
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=15):
        """Add a bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.cell(indent, 5.5, '*', new_x="END")
        self.multi_cell(0, 5.5, f" {text}")
        self.ln(1)

    def code_block(self, text):
        """Add a code block with background."""
        self.set_fill_color(240, 240, 240)
        self.set_font("Courier", "", 9)
        self.set_text_color(40, 40, 40)
        lines = text.split("\n")
        for line in lines:
            self.cell(0, 5, f"  {line}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def highlight_box(self, text, color="blue"):
        """Add a highlighted info box."""
        if color == "blue":
            self.set_fill_color(235, 245, 255)
            self.set_draw_color(41, 128, 185)
        elif color == "green":
            self.set_fill_color(235, 255, 240)
            self.set_draw_color(39, 174, 96)
        elif color == "orange":
            self.set_fill_color(255, 248, 235)
            self.set_draw_color(243, 156, 18)
        elif color == "red":
            self.set_fill_color(255, 235, 235)
            self.set_draw_color(231, 76, 60)

        self.set_line_width(0.4)
        x, y = self.get_x(), self.get_y()
        self.set_font("Helvetica", "I", 9.5)
        self.set_text_color(50, 50, 50)

        # Calculate height
        w = self.w - 2 * self.l_margin - 6
        lines = self.multi_cell(w, 5, text, split_only=True)
        h = len(lines) * 5 + 6

        if self.get_y() + h > self.h - 20:
            self.add_page()
            y = self.get_y()

        self.rect(x, y, self.w - 2 * self.l_margin, h, style="DF")
        self.set_xy(x + 3, y + 3)
        self.multi_cell(w, 5, text)
        self.ln(4)

    def simple_table(self, headers, data, col_widths=None):
        """Draw a simple table."""
        if col_widths is None:
            n = len(headers)
            col_widths = [(self.w - 2 * self.l_margin) / n] * n

        # Check if table fits on page
        estimated_h = (len(data) + 1) * 7 + 5
        if self.get_y() + estimated_h > self.h - 25:
            self.add_page()

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, str(val), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def build_report():
    """Build the full PDF report."""
    pdf = PipelineReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ╔══════════════════════════════════════════╗
    #  COVER PAGE
    # ╚══════════════════════════════════════════╝
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 15, "Systeme de Prediction", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "de Defaut de Paiement", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Rapport Complet du Pipeline de Machine Learning", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Auteur : Yomna Haouel", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Projet : Data Science  -  Credit Scoring", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Dataset : Home Credit Default Risk", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Pipeline complet : EDA, Preprocessing, Feature Engineering,", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Feature Selection, Multi-Model Training, MLflow, SHAP, FastAPI", align="C", new_x="LMARGIN", new_y="NEXT")

    # ╔══════════════════════════════════════════╗
    #  TABLE OF CONTENTS
    # ╚══════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("Table des Matieres")
    pdf.ln(2)

    toc = [
        "1. Vue d'Ensemble de l'Architecture",
        "2. Le Dataset  -  Home Credit Default Risk",
        "3. Etape 0  -  Analyse Exploratoire (EDA)",
        "4. Etape 1  -  Preprocessing Anti-Fuite",
        "5. Etape 2  -  Feature Engineering",
        "6. Etape 3  -  Selection de Variables",
        "7. Etape 4  -  Entrainement Multi-Modeles",
        "8. Etape 5  -  Comparaison et Selection du Meilleur Modele",
        "9. Deploiement  -  API FastAPI",
        "10. Bonnes Pratiques Implementees",
        "11. Instructions d'Execution",
        "12. Points Cles pour la Soutenance",
    ]
    for item in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ╔══════════════════════════════════════════╗
    #  1. ARCHITECTURE
    # ╚══════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("1. Vue d'Ensemble de l'Architecture")

    pdf.body_text(
        "Le projet suit une architecture modulaire industrielle : chaque etape du pipeline "
        "est isolee dans un fichier Python dedie, et tous les parametres sont centralises dans "
        "un fichier de configuration unique (config.py). L'orchestration est assuree par "
        "code/main.py qui enchaine les 5 etapes dans l'ordre."
    )

    pdf.section_title("Structure des fichiers", level=3)
    pdf.code_block(
        "config.py                  -> Configuration centralisee\n"
        "code/main.py               -> Orchestrateur principal (CLI)\n"
        "code/exploration_donnees.py -> Etape 0 : EDA\n"
        "code/preprocessing.py      -> Etape 1 : Nettoyage anti-fuite\n"
        "code/feature_engineering.py -> Etape 2 : Creation de variables\n"
        "code/feature_selection.py   -> Etape 3 : Selection de variables\n"
        "code/training.py           -> Etape 4 : Entrainement + MLflow\n"
        "code/model_comparison.py   -> Etape 5 : Comparaison + SHAP\n"
        "api/app.py                 -> Deploiement FastAPI"
    )

    pdf.section_title("Pourquoi cette architecture ?", level=3)
    pdf.bullet("Reproductibilite : tout parametre est dans config.py, jamais en dur dans le code")
    pdf.bullet("Maintenabilite : modifier une etape n'affecte pas les autres")
    pdf.bullet("Testabilite : chaque module peut etre execute/teste independamment")
    pdf.bullet("Tracabilite : MLflow enregistre chaque experience")

    # ╔══════════════════════════════════════════╗
    #  2. DATASET
    # ╚══════════════════════════════════════════╝
    pdf.section_title("2. Le Dataset  -  Home Credit Default Risk")

    pdf.simple_table(
        ["Caracteristique", "Valeur"],
        [
            ["Fichier", "feature_matrix.csv"],
            ["Dimensions", "356 255 lignes, 1 698 colonnes"],
            ["Train", "307 511 lignes (set = 'train')"],
            ["Test", "48 744 lignes (set = 'test')"],
            ["Variable cible", "TARGET : 0=bon payeur, 1=defaut"],
            ["Desequilibre", "~8% de defauts (classe 1)"],
        ],
        col_widths=[60, 130],
    )

    pdf.body_text(
        "Contexte metier : Une institution financiere doit decider si un client va rembourser "
        "son credit. Le cout d'un defaut non detecte (faux negatif) est bien superieur au cout "
        "d'un refus a tort (faux positif). C'est pourquoi le Recall est une metrique prioritaire."
    )

    pdf.highlight_box(
        "Le dataset est fortement desequilibre (~8% de defauts). "
        "Sans traitement adapte, un modele naif predisant toujours '0' obtiendrait 92% "
        "d'accuracy mais 0% de Recall  -  completement inutile en pratique.", "orange"
    )

    # ╔══════════════════════════════════════════╗
    #  3. EDA
    # ╚══════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("3. Etape 0  -  Analyse Exploratoire (EDA)")
    pdf.bold_text("Fichier : code/exploration_donnees.py")
    pdf.bold_text("Commande : python code/main.py --eda-only")
    pdf.ln(2)

    pdf.section_title("3.1 Distribution de la cible", level=2)
    pdf.body_text(
        "La fonction analyze_target() calcule le taux de defaut (~8.07%) et genere un "
        "diagramme en barres. Cette analyse revele le desequilibre des classes qui necessitera "
        "un traitement special (SMOTE) lors de l'entrainement."
    )

    pdf.section_title("3.2 Analyse des valeurs manquantes", level=2)
    pdf.body_text(
        "La fonction analyze_missing() identifie les colonnes avec plus de 50% de valeurs "
        "manquantes et genere un histogramme des taux de missing. Cette analyse justifie "
        "le seuil de suppression a 60% utilise dans le preprocessing."
    )

    pdf.section_title("3.3 Distributions numeriques", level=2)
    pdf.body_text(
        "La fonction analyze_numeric() trace les histogrammes des 20 variables numeriques "
        "les plus correlees avec TARGET, separees par classe (defaut vs non-defaut). "
        "Cela permet d'identifier visuellement les variables discriminantes."
    )

    pdf.section_title("3.4 Detection d'outliers", level=2)
    pdf.body_text(
        "Methode IQR (InterQuartile Range) : un point est considere comme outlier "
        "s'il est en dehors de [Q1 - 1.5 x IQR, Q3 + 1.5 x IQR]. La fonction "
        "detect_outliers() identifie les colonnes les plus polluees par les valeurs aberrantes "
        "et genere un Top 20 des colonnes avec le plus d'outliers."
    )

    pdf.section_title("3.5 Analyse categorielle", level=2)
    pdf.body_text(
        "Distribution des variables categorielles et taux de defaut par modalite. "
        "Tres utile pour des variables comme NAME_CONTRACT_TYPE, CODE_GENDER, etc."
    )

    pdf.section_title("3.6 Correlation avec la cible", level=2)
    pdf.body_text(
        "Top 20 des variables les plus correlees avec TARGET + heatmap de correlation. "
        "Donne un premier apercu des variables importantes (EXT_SOURCE_1/2/3, DAYS_BIRTH, etc.)."
    )

    pdf.section_title("3.7 Recommandations business", level=2)
    pdf.body_text(
        "5 insights automatiques generes a la fin de l'EDA : gestion des missing, traitement "
        "du desequilibre, variables cles, feature engineering suggere, strategie de modelisation."
    )

    pdf.highlight_box(
        "Pourquoi l'EDA est essentielle ? L'EDA guide toutes les decisions en aval : quels seuils "
        "de nettoyage choisir, quelles variables transformer, quel algorithme privilegier. "
        "C'est la fondation scientifique du projet.", "blue"
    )

    # ╔══════════════════════════════════════════════════╗
    #  4. PREPROCESSING
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("4. Etape 1  -  Preprocessing Anti-Fuite")
    pdf.bold_text("Fichier : code/preprocessing.py  -  Classe : DataPreprocessor")
    pdf.ln(2)

    pdf.section_title("4.1 Chargement et filtrage", level=2)
    pdf.body_text(
        "La methode load_and_filter() garde uniquement les lignes d'entrainement "
        "(set == 'train'), supprime les colonnes d'identification (SK_ID_CURR, set), "
        "et separe X (features) et y (TARGET)."
    )

    pdf.section_title("4.2 Nettoyage basique", level=2)
    pdf.body_text(
        "La methode basic_cleaning() supprime les colonnes avec plus de 60% de valeurs "
        "manquantes (829 colonnes supprimees sur 1 695) et supprime les colonnes a variance "
        "nulle (constantes). Resultat : 867 colonnes restantes."
    )

    pdf.section_title("4.3 Split stratifie", level=2)
    pdf.body_text(
        "train_test_split(test_size=0.2, stratify=y, random_state=42) produit "
        "246 008 echantillons d'entrainement et 61 503 de validation. Le split stratifie "
        "preserve le ratio ~8% de defauts dans les deux ensembles."
    )

    pdf.section_title("4.4 Pipeline sklearn (ColumnTransformer)", level=2)
    pdf.body_text(
        "Un ColumnTransformer avec deux branches paralleles :"
    )
    pdf.bullet("Numeriques : SimpleImputer(strategy='median') puis StandardScaler()")
    pdf.bullet("Categorielles : SimpleImputer(strategy='most_frequent') puis OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)")

    pdf.section_title("4.5 Fit-Transform anti-fuite", level=2)
    pdf.body_text(
        "fit_transform() est applique sur le train UNIQUEMENT. transform() est applique "
        "sur le test (jamais de fit sur le test !). Le pipeline est sauvegarde "
        "(preprocessing_pipeline.joblib) pour le deploiement en production."
    )

    pdf.highlight_box(
        "POURQUOI L'ANTI-FUITE EST CRITIQUE ?\n"
        "La fuite de donnees (data leakage) est l'erreur la plus grave en ML : si le modele "
        "'voit' des informations du test pendant l'entrainement, les metriques sont "
        "artificiellement gonflees et le modele ne generalisera pas en production.\n\n"
        "Notre pipeline garantit ZERO fuite :\n"
        "- Le split est fait AVANT toute transformation\n"
        "- Le StandardScaler apprend la moyenne/ecart-type sur le train uniquement\n"
        "- L'OrdinalEncoder apprend les categories sur le train uniquement\n"
        "- Le SMOTE (etape 4) est applique sur le train uniquement", "red"
    )

    # ╔══════════════════════════════════════════════════╗
    #  5. FEATURE ENGINEERING
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("5. Etape 2  -  Feature Engineering")
    pdf.bold_text("Fichier : code/feature_engineering.py  -  Classe : FeatureEngineer")
    pdf.bold_text("20 nouvelles variables creees, regroupees en 5 familles :")
    pdf.ln(2)

    pdf.section_title("5.1 Ratios financiers (6 variables)", level=2)
    pdf.simple_table(
        ["Variable", "Formule", "Logique Metier"],
        [
            ["CREDIT_INCOME_RATIO", "Credit / Revenu", "Capacite de remboursement"],
            ["ANNUITY_INCOME_RATIO", "Annuite / Revenu", "Poids des mensualites"],
            ["CREDIT_TERM", "Credit / Annuite", "Duree estimee du credit"],
            ["CREDIT_GOODS_RATIO", "Credit / Prix du bien", "Surcout du credit"],
            ["INCOME_PER_PERSON", "Revenu / Nb famille", "Revenu par personne"],
            ["ANNUITY_CREDIT_RATIO", "Annuite / Credit", "Taux de remboursement"],
        ],
        col_widths=[55, 45, 90],
    )

    pdf.section_title("5.2 Age et Emploi (4 variables)", level=2)
    pdf.simple_table(
        ["Variable", "Description"],
        [
            ["AGE_YEARS", "Age en annees (DAYS_BIRTH / -365)"],
            ["EMPLOYED_ANOMALY", "Flag binaire : DAYS_EMPLOYED == 365243 (anomalie connue)"],
            ["EMPLOYMENT_YEARS", "Anciennete en annees (DAYS_EMPLOYED / -365)"],
            ["EMPLOYMENT_AGE_RATIO", "Ratio anciennete / age"],
        ],
        col_widths=[55, 135],
    )
    pdf.body_text(
        "La valeur 365243 dans DAYS_EMPLOYED est une anomalie connue du dataset Home Credit "
        "(retraites, chomeurs). Un flag binaire EMPLOYED_ANOMALY est cree pour la capturer "
        "explicitement."
    )

    pdf.section_title("5.3 Interactions EXT_SOURCE (7 variables)", level=2)
    pdf.body_text(
        "Les scores externes (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) proviennent de "
        "bureaux de credit et sont les variables les plus predictives du dataset. "
        "Les interactions creees sont :"
    )
    pdf.bullet("Statistiques agregees : EXT_SOURCE_MEAN, EXT_SOURCE_STD, EXT_SOURCE_MIN, EXT_SOURCE_MAX")
    pdf.bullet("Produits croises : EXT1 x EXT2, EXT1 x EXT3, EXT2 x EXT3")
    pdf.body_text(
        "Ces interactions capturent des relations non-lineaires que les modeles "
        "lineaires ne capturent pas seuls."
    )

    pdf.section_title("5.4 Documents fournis (1 variable)", level=2)
    pdf.body_text(
        "DOCUMENTS_PROVIDED_COUNT = somme des 20 flags FLAG_DOCUMENT_*. Un client qui "
        "fournit plus de documents est souvent plus solvable."
    )

    pdf.section_title("5.5 Variables sociales (2 variables)", level=2)
    pdf.bullet("REGION_MISMATCH_SUM : somme des incoherences geographiques (client/employeur/contact)")
    pdf.bullet("CONTACT_INFO_COUNT : nombre de moyens de contact fournis (signaux de fraude potentielle)")

    pdf.highlight_box(
        "Pourquoi le Feature Engineering est important ?\n"
        "- Les algorithmes ML travaillent mieux avec des combinaisons significatives "
        "qu'avec des variables brutes\n"
        "- CREDIT_INCOME_RATIO est bien plus discriminant que AMT_CREDIT et AMT_INCOME_TOTAL "
        "pris separement\n"
        "- Les interactions EXT_SOURCE capturent des synergies entre scores de credit\n"
        "- Gestion robuste : verification d'existence des colonnes sources, remplacement "
        "des divisions par zero", "green"
    )

    # ╔══════════════════════════════════════════════════╗
    #  6. FEATURE SELECTION
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("6. Etape 3  -  Selection de Variables")
    pdf.bold_text("Fichier : code/feature_selection.py  -  Classe : FeatureSelector")
    pdf.bold_text("Processus en 3 phases + strategie de fusion")
    pdf.ln(2)

    pdf.section_title("Phase 1  -  Filtre de correlation", level=2)
    pdf.body_text(
        "Calcule la matrice de correlation de Pearson (sur un echantillon de 50K lignes "
        "pour la performance). Si deux variables ont |correlation| > 0.85, celle avec la "
        "plus faible correlation avec TARGET est supprimee."
    )
    pdf.bold_text("Resultat : 850 -> 449 variables (401 redondantes supprimees)")
    pdf.italic_text(
        "Pourquoi ? La multicolinearite degrade les modeles lineaires et alourdit "
        "les modeles d'ensemble sans gain predictif."
    )

    pdf.section_title("Phase 2  -  Information Mutuelle", level=2)
    pdf.body_text(
        "Calcule le MI score de chaque variable avec TARGET (sur un echantillon de 80K). "
        "Garde les top 120 variables (parametre MI_TOP_K). "
        "L'information mutuelle capture les dependances NON-LINEAIRES, contrairement "
        "a la correlation de Pearson."
    )
    pdf.italic_text(
        "Pourquoi ? Identifie des variables importantes que la correlation lineaire "
        "pourrait manquer."
    )

    pdf.section_title("Phase 3  -  Importance par arbre (LightGBM)", level=2)
    pdf.body_text(
        "Entraine un LightGBM rapide et extrait l'importance des variables "
        "(feature_importances_). Garde les variables dont l'importance normalisee > 0.001. "
        "230 variables retenues."
    )
    pdf.italic_text(
        "Pourquoi ? Les modeles d'ensemble evaluent l'utilite reelle de chaque variable "
        "dans des splits de decision."
    )

    pdf.section_title("Fusion  -  Strategie Union", level=2)
    pdf.body_text(
        "Notre choix : la strategie UNION garde toute variable selectionnee par AU MOINS "
        "une methode. L'alternative 'intersection' ne garderait que les variables selectionnees "
        "par TOUTES les methodes (plus restrictif)."
    )

    pdf.highlight_box(
        "Resultat final : de 886 variables -> 269 variables finales (reduction de 70%)\n\n"
        "Pourquoi union ? Chaque methode a des forces differentes (lineaire vs non-lineaire "
        "vs importance d'arbre). L'union maximise les chances de garder les bonnes variables. "
        "Les features selectionnees sont sauvegardees dans selected_features.json pour "
        "le deploiement.", "blue"
    )

    # ╔══════════════════════════════════════════════════╗
    #  7. TRAINING
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("7. Etape 4  -  Entrainement Multi-Modeles")
    pdf.bold_text("Fichier : code/training.py")
    pdf.ln(2)

    pdf.section_title("7.1 SMOTE (Synthetic Minority Oversampling)", level=2)
    pdf.body_text(
        "Le dataset est desequilibre (8% de defauts). Sans traitement, les modeles predisent "
        "'0' tout le temps et obtiennent 92% d'accuracy... mais 0% de Recall !"
    )
    pdf.body_text(
        "SMOTE genere des echantillons synthetiques de la classe minoritaire par interpolation "
        "entre voisins KNN. Avec sampling_strategy=0.5, la classe minoritaire atteint 50% "
        "de la classe majoritaire apres SMOTE (ratio 1:2 au lieu de 1:12)."
    )
    pdf.highlight_box(
        "SMOTE est applique sur le train UNIQUEMENT  -  jamais sur le test. "
        "C'est une regle fondamentale de l'anti-leakage.", "red"
    )

    pdf.section_title("7.2 Les 5 modeles entraines", level=2)
    pdf.simple_table(
        ["Modele", "Type", "Particularite"],
        [
            ["LogisticRegression", "Lineaire", "class_weight='balanced', C=0.01"],
            ["RandomForest", "Bagging", "200 arbres, prof. 12, balanced"],
            ["XGBoost", "Boosting", "300 arbres, lr=0.03, scale_pos_weight=10"],
            ["LightGBM", "Boosting", "500 arbres, lr=0.03, is_unbalance=True"],
            ["CatBoost", "Boosting", "500 iter, lr=0.03, auto_class_weights"],
        ],
        col_widths=[40, 30, 120],
    )

    pdf.section_title("7.3 Metriques d'evaluation", level=2)
    pdf.body_text("Chaque modele est evalue sur le jeu de validation (jamais vu pendant l'entrainement) :")
    pdf.bullet("ROC-AUC : mesure la capacite discriminante globale du modele (independante du seuil)")
    pdf.bullet("Recall : proportion de vrais defauts detectes (priorite business)")
    pdf.bullet("Precision : proportion de predictions 'defaut' correctes")
    pdf.bullet("F1-Score : moyenne harmonique precision/recall")
    pdf.body_text(
        "Pour chaque modele, une matrice de confusion et une courbe ROC sont "
        "generees et sauvegardees dans plots/training/."
    )

    pdf.section_title("7.4 Tracking MLflow", level=2)
    pdf.body_text(
        "Chaque run MLflow enregistre : les parametres du modele, les metriques de performance, "
        "et le modele serialise. Cela permet de comparer les experiences a posteriori, "
        "de retrouver les meilleurs hyperparametres, et de reproduire n'importe quel resultat."
    )

    # ╔══════════════════════════════════════════════════╗
    #  8. COMPARISON
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("8. Etape 5  -  Comparaison et Selection")
    pdf.bold_text("Fichier : code/model_comparison.py")
    pdf.ln(2)

    pdf.section_title("8.1 Tableau comparatif des resultats", level=2)
    pdf.simple_table(
        ["Modele", "ROC-AUC", "Recall", "F1-Score", "Temps (s)"],
        [
            ["LightGBM", "0.7771", "0.1253", "0.1947", "29.8"],
            ["CatBoost", "0.7738", "0.1148", "0.1815", "81.2"],
            ["XGBoost", "0.7583", "0.7140", "0.2598", "25.1"],
            ["RandomForest", "0.7375", "0.1732", "0.2125", "107.0"],
            ["LogisticRegression", "0.6519", "0.6137", "0.2034", "78.7"],
        ],
        col_widths=[42, 30, 30, 30, 30],
    )

    pdf.section_title("8.2 Critere de selection pondere", level=2)
    pdf.body_text("La formule de selection du meilleur modele est :")
    pdf.ln(1)
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 8, "Score = 0.50 x AUC + 0.30 x Recall + 0.20 x F1", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.body_text(
        "Ponderation :\n"
        "- 50% AUC : pouvoir discriminant global\n"
        "- 30% Recall : detection des defauts (objectif metier prioritaire)\n"
        "- 20% F1 : equilibre precision/recall"
    )

    pdf.highlight_box(
        "RESULTAT : XGBoost gagne avec un score pondere de 0.6453.\n\n"
        "Justification : XGBoost offre le meilleur compromis AUC (0.7583) + Recall eleve "
        "(0.7140). LightGBM a un meilleur AUC (0.7771) mais un Recall tres faible (0.1253), "
        "ce qui signifie qu'il ne detecte que 1 defaut sur 8  -  inacceptable en contexte "
        "bancaire ou le cout d'un defaut non detecte est tres superieur au cout d'un "
        "faux positif.", "green"
    )

    pdf.section_title("8.3 Analyse SHAP (Interpretabilite)", level=2)
    pdf.body_text(
        "TreeExplainer de SHAP calcule la contribution marginale de chaque variable a chaque "
        "prediction. Le summary plot montre les 20 variables les plus influentes du meilleur "
        "modele (XGBoost). Cette interpretabilite est indispensable en contexte bancaire "
        "pour la conformite reglementaire et la transparence des decisions de credit."
    )

    # ╔══════════════════════════════════════════════════╗
    #  9. FASTAPI
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("9. Deploiement  -  API FastAPI")
    pdf.bold_text("Fichier : api/app.py")
    pdf.ln(2)

    pdf.body_text(
        "L'API REST permet de deployer le modele en production pour des predictions "
        "en temps reel. Elle charge au demarrage le meilleur modele (best_model.joblib), "
        "la liste des features (selected_features.json), et le pipeline de preprocessing."
    )

    pdf.section_title("Endpoints disponibles", level=2)
    pdf.simple_table(
        ["Endpoint", "Methode", "Description"],
        [
            ["/", "GET", "Health check  -  verifie que le modele est charge"],
            ["/info", "GET", "Informations du modele et features attendues"],
            ["/predict", "POST", "Prediction pour un client unique"],
        ],
        col_widths=[40, 30, 120],
    )

    pdf.section_title("Classification du risque", level=2)
    pdf.simple_table(
        ["Probabilite", "Niveau de Risque"],
        [
            ["< 0.2", "LOW (Faible)"],
            ["0.2 a 0.5", "MEDIUM (Moyen)"],
            ["0.5 a 0.8", "HIGH (Eleve)"],
            [">= 0.8", "VERY HIGH (Tres Eleve)"],
        ],
        col_widths=[60, 130],
    )

    pdf.section_title("Exemple d'utilisation", level=2)
    pdf.code_block(
        '# Lancer le serveur :\n'
        'uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload\n'
        '\n'
        '# Envoyer une requete de prediction :\n'
        'curl -X POST http://localhost:8000/predict \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"features": {"AMT_CREDIT": 500000, ...}}\''
    )

    pdf.body_text(
        "La reponse contient : la probabilite de defaut, la prediction binaire (0 ou 1), "
        "le niveau de risque (LOW/MEDIUM/HIGH/VERY HIGH), et le seuil utilise (0.5)."
    )

    # ╔══════════════════════════════════════════════════╗
    #  10. BONNES PRATIQUES
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("10. Bonnes Pratiques Implementees")
    pdf.ln(2)

    pdf.simple_table(
        ["Pratique", "Implementation", "Justification"],
        [
            ["Anti data leakage", "Split avant transformation", "Metriques fiables"],
            ["Config centralisee", "Tout dans config.py", "Reproductibilite"],
            ["Architecture modulaire", "1 fichier = 1 responsabilite", "Maintenabilite"],
            ["SMOTE train only", "Jamais sur le test", "Pas de contamination"],
            ["Selection multi-methodes", "Corr + MI + Tree", "Robustesse"],
            ["Tracking MLflow", "Params + metriques + modele", "Tracabilite"],
            ["Seuils ponderes", "AUC/Recall/F1", "Choix objectif"],
            ["SHAP", "Interpretabilite post-hoc", "Conformite reglementaire"],
            ["API REST", "FastAPI + Pydantic", "Production-ready"],
        ],
        col_widths=[48, 55, 87],
    )

    # ╔══════════════════════════════════════════════════╗
    #  11. INSTRUCTIONS
    # ╚══════════════════════════════════════════════════╝
    pdf.section_title("11. Instructions d'Execution")
    pdf.ln(2)

    pdf.section_title("Installation", level=2)
    pdf.code_block("pip install -r requirements.txt")

    pdf.section_title("Lancer l'EDA seule", level=2)
    pdf.code_block("python code/main.py --eda-only")

    pdf.section_title("Lancer le pipeline complet (avec EDA)", level=2)
    pdf.code_block("python code/main.py --eda")

    pdf.section_title("Lancer le pipeline complet (sans EDA, plus rapide)", level=2)
    pdf.code_block("python code/main.py")

    pdf.section_title("Lancer l'API de prediction", level=2)
    pdf.code_block("uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")

    pdf.section_title("Visualiser les experiences MLflow", level=2)
    pdf.code_block("mlflow ui --port 5000")

    # ╔══════════════════════════════════════════════════╗
    #  12. SOUTENANCE
    # ╚══════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("12. Points Cles pour la Soutenance")
    pdf.ln(2)

    pdf.section_title("Question : Pourquoi XGBoost est le meilleur modele ?", level=2)
    pdf.highlight_box(
        "XGBoost offre le meilleur compromis entre AUC (0.7583) et Recall (0.7140). "
        "Bien que LightGBM ait un AUC superieur (0.7771), son Recall de 12.5% signifie "
        "qu'il ne detecte que 1 defaut sur 8  -  ce qui est inacceptable dans un contexte "
        "de risque de credit ou le cout d'un defaut non detecte est tres superieur au "
        "cout d'un faux positif.", "blue"
    )

    pdf.section_title("Question : Comment avez-vous gere le desequilibre ?", level=2)
    pdf.highlight_box(
        "Trois niveaux :\n"
        "1) SMOTE pour reequilibrer le train a un ratio 1:2\n"
        "2) Hyperparametres natifs (scale_pos_weight, class_weight='balanced', is_unbalance)\n"
        "3) Optimisation sur le Recall et l'AUC plutot que l'accuracy", "blue"
    )

    pdf.section_title("Question : Comment garantissez-vous l'absence de data leakage ?", level=2)
    pdf.highlight_box(
        "Le split intervient AVANT toute transformation. Le pipeline sklearn est fit sur "
        "le train uniquement puis transform sur le test. Le SMOTE est applique au train "
        "uniquement. Aucune information du test ne 'fuit' dans l'entrainement.", "blue"
    )

    pdf.section_title("Question : Pourquoi 3 methodes de selection de variables ?", level=2)
    pdf.highlight_box(
        "Chaque methode capture des aspects differents :\n"
        "- La correlation filtre la redondance lineaire\n"
        "- L'information mutuelle detecte les dependances non-lineaires\n"
        "- L'importance d'arbre evalue l'utilite reelle dans un modele de decision\n\n"
        "L'union des trois garantit qu'aucune variable utile n'est perdue.", "blue"
    )

    pdf.section_title("Question : Pourquoi utiliser MLflow ?", level=2)
    pdf.highlight_box(
        "MLflow assure la tracabilite complete des experiences : chaque run enregistre "
        "les hyperparametres, les metriques et le modele serialise. Cela permet de "
        "reproduire n'importe quel resultat, de comparer les experiences a posteriori, "
        "et de deployer facilement le meilleur modele en production.", "blue"
    )

    pdf.section_title("Question : Comment le modele est-il deploye ?", level=2)
    pdf.highlight_box(
        "Via une API REST FastAPI. Au demarrage, l'API charge le best_model.joblib "
        "et le selected_features.json. Un endpoint POST /predict recoit les donnees "
        "client en JSON, aligne les features, effectue la prediction, et retourne "
        "la probabilite de defaut avec un niveau de risque (LOW/MEDIUM/HIGH/VERY HIGH).", "blue"
    )

    # ── Save ──
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Rapport_Pipeline_Credit_Scoring.pdf"
    )
    pdf.output(output_path)
    print(f"\n{'=' * 60}")
    print(f"  PDF genere avec succes !")
    print(f"  -> {output_path}")
    print(f"{'=' * 60}")
    return output_path


if __name__ == "__main__":
    build_report()
