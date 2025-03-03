import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Plot:
    """
    Classe pour visualiser une variable quantitative ou catégorielle avec des graphiques adaptés.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialise le visualiseur avec un DataFrame.
        :param data: DataFrame contenant les données.
        """
        self.data = data

    def histogram(self, var_quant, var_cat=None, bins=30):
        """
        Affiche un histogramme pour une variable quantitative.
        :param var_quant: Nom de la variable quantitative.
        :param var_cat: (Optionnel) Nom de la variable catégorielle pour comparaison.
        :param bins: Nombre de bins.
        """
        plt.figure(figsize=(8, 5))
        if var_cat:
            sns.histplot(data=self.data, x=var_quant, hue=var_cat, bins=bins, kde=True, palette="Set2", alpha=0.7)
        else:
            sns.histplot(self.data[var_quant], bins=bins, kde=True, color="blue", alpha=0.7)
        plt.title(f"Histogramme de {var_quant}")
        plt.xlabel(var_quant)
        plt.grid(True)
        plt.show()

    def density(self, var_quant, var_cat=None):
        """
        Affiche une courbe de densité pour une variable quantitative.
        :param var_quant: Nom de la variable quantitative.
        :param var_cat: (Optionnel) Nom de la variable catégorielle pour comparaison.
        """
        plt.figure(figsize=(8, 5))
        if var_cat:
            sns.kdeplot(data=self.data, x=var_quant, hue=var_cat, fill=True, common_norm=False, palette="Set2")
        else:
            sns.kdeplot(self.data[var_quant], fill=True, color="blue")
        plt.title(f"Densité de {var_quant}")
        plt.xlabel(var_quant)
        plt.grid(True)
        plt.show()

    def boxplot(self, var_quant, var_cat=None):
        """
        Affiche un boxplot pour une variable quantitative.
        :param var_quant: Nom de la variable quantitative.
        :param var_cat: (Optionnel) Nom de la variable catégorielle pour comparaison.
        """
        plt.figure(figsize=(8, 5))
        if var_cat:
            sns.boxplot(data=self.data, x=var_cat, y=var_quant, palette="Set2")
        else:
            sns.boxplot(y=self.data[var_quant], color="blue")
        plt.title(f"Boxplot de {var_quant}")
        plt.grid(True)
        plt.show()

    def violin(self, var_quant, var_cat=None):
        """
        Affiche un violin plot pour une variable quantitative.
        :param var_quant: Nom de la variable quantitative.
        :param var_cat: (Optionnel) Nom de la variable catégorielle pour comparaison.
        """
        plt.figure(figsize=(8, 5))
        if var_cat:
            sns.violinplot(data=self.data, x=var_cat, y=var_quant, palette="Set2")
        else:
            sns.violinplot(y=self.data[var_quant], color="blue")
        plt.title(f"Violin plot de {var_quant}")
        plt.grid(True)
        plt.show()

    def barplot(self, var_cat):
        """
        Affiche un barplot pour une variable catégorielle.
        :param var_cat: Nom de la variable catégorielle.
        """
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.data, x=var_cat, palette="Set2")
        plt.title(f"Répartition des catégories de {var_cat}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def scatter(self, var_x, var_y, var_cat=None):
        """
        Affiche un scatterplot entre deux variables quantitatives.
        :param var_x: Nom de la variable en axe X.
        :param var_y: Nom de la variable en axe Y.
        :param var_cat: (Optionnel) Nom de la variable catégorielle pour colorer les points.
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.data, x=var_x, y=var_y, hue=var_cat, palette="Set2", alpha=0.8)
        plt.title(f"Scatterplot de {var_x} vs {var_y}")
        plt.grid(True)
        plt.show()

# ====== Exemple d'utilisation ======
if __name__ == "__main__":
    # Création d'un DataFrame exemple
    df = pd.DataFrame({
        "Revenu": [2500, 3000, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000],
        "Âge": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Catégorie": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"]
    })

    # Instanciation et affichage des graphiques
    visualizer = Plot(df)
    visualizer.histogram("Revenu")
    visualizer.density("Revenu", var_cat="Catégorie")
    visualizer.boxplot("Revenu", var_cat="Catégorie")
    visualizer.violin("Revenu", var_cat="Catégorie")
    visualizer.barplot("Catégorie")
    visualizer.scatter("Âge", "Revenu", var_cat="Catégorie")
