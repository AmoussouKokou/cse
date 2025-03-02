# ----------------- Installation de packages ------------------------------

print("------------ PACKAGES ------------")
packages <- c("dplyr", "ggplot2", "tidyr", "readr", "ggpubr", "readxl")

# Fonction pour installer les packages manquants
packages_to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(packages_to_install) > 0) {
  install.packages(packages_to_install)
}

# Charger tous les packages
lapply(packages, library, character.only = TRUE)


# ----------------- Importation des données -----------------
print("------------ Imports ------------") 
train <- read_csv("data/raw/cs-training.csv") %>% select(-1)
test <- read_csv("data/raw/cs-test.csv") %>% select(-1)
data_dict <- read_excel("data/raw/Data Dictionary.xls")

# ----------------- Data Dictionary --------------------------------

print("------------ Data Dictionary ------------")
# Ajout des détails des variables en francais
descriptions <- c(
  "Personne ayant un retard de paiement de 90 jours ou plus",
  "Solde total des cartes de crédit et des lignes de crédit personnelles, à l'exception des biens immobiliers et des dettes à tempérament telles que les prêts automobiles, divisé par la somme des limites de crédit",
  "Âge de l'emprunteur en années",
  "Nombre de fois où l'emprunteur a été en retard de paiement de 30 à 59 jours, mais pas plus, au cours des deux dernières années",
  "Paiements mensuels des dettes, pensions alimentaires, frais de subsistance divisés par le revenu mensuel brut",
  "Revenu mensuel",
  "Nombre de prêts ouverts (à tempérament comme un prêt automobile ou un prêt hypothécaire) et de lignes de crédit (par exemple, cartes de crédit)",
  "Nombre de fois où l'emprunteur a été en retard de 90 jours ou plus",
  "Nombre de prêts hypothécaires et immobiliers, y compris les lignes de crédit immobilier",
  "Nombre de fois où l'emprunteur a été en retard de 60 à 89 jours, mais pas plus au cours des deux dernières années",
  "Nombre de personnes à charge dans la famille, à l'exclusion de l'emprunteur lui-même (conjoint, enfants, etc.)"
)
# names(xtrain)
#  [1] "SeriousDlqin2yrs"
#  [2] "RevolvingUtilizationOfUnsecuredLines"
#  [3] "age"
#  [4] "NumberOfTime30-59DaysPastDueNotWorse"
#  [5] "DebtRatio"
#  [6] "MonthlyIncome"
#  [7] "NumberOfOpenCreditLinesAndLoans"
#  [8] "NumberOfTimes90DaysLate"
#  [9] "NumberRealEstateLoansOrLines"
# [10] "NumberOfTime60-89DaysPastDueNotWorse"
# [11] "NumberOfDependents"

data_dict = data_dict %>% mutate(descriptions_fr = descriptions)

# Export

write_csv(data_dict, "data/processed/dictionary.csv")


# ----------------- Train data -----------------
noms <- names(train)

print("------------ Train data ------------")
## Variable cible : 
x <- train %>% select(1) %>% pull()
x_name <- noms[1]
print(x_name)