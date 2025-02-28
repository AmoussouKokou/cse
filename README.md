# cse
Credit Score Evaluator. Développer un score de crédit : créer un modèle de machine learning pour évaluer les scores de crédit.



Voici une explication détaillée de chacune de ces variables dans le cadre d’un **score de risque** :

---

### 1. **Personne ayant un retard de paiement de 90 jours ou plus**  
Cette variable représente une **caractéristique de l'historique de paiement** de l'emprunteur. Elle indique si l'emprunteur a déjà eu un retard de paiement important, spécifiquement supérieur ou égal à **90 jours**. Un tel retard est souvent un indicateur de **difficultés financières importantes**, et cela peut augmenter le risque pour le prêteur. Si l'emprunteur a eu un retard de 90 jours ou plus dans le passé, il est considéré comme présentant un risque élevé.

---

### 2. **Solde total des cartes de crédit et des lignes de crédit personnelles, à l'exception des biens immobiliers et des dettes à tempérament telles que les prêts automobiles, divisé par la somme des limites de crédit**  
Cette variable calcule le **ratio d'utilisation du crédit**. Il s'agit de la **part du crédit utilisé** par rapport au crédit disponible. Un ratio élevé signifie que l'emprunteur utilise une grande proportion de ses lignes de crédit disponibles, ce qui peut être un indicateur de **dépendance au crédit** et de **risque financier**. Un tel comportement peut signifier que l'emprunteur est proche de ses limites financières, ce qui peut augmenter son risque de défaut de paiement.

---

### 3. **Âge de l'emprunteur en années**  
L'âge de l'emprunteur est une donnée démographique qui peut avoir un impact sur son **capacité à rembourser les dettes**. En général :
- Les jeunes emprunteurs peuvent être perçus comme moins expérimentés financièrement, ce qui peut entraîner un risque plus élevé.
- Les emprunteurs plus âgés (en particulier ceux qui approchent de la retraite) peuvent avoir un revenu moins stable, ce qui peut aussi influencer leur **capacité à rembourser**.
Cette variable permet de mieux comprendre la **stabilité financière** de l'emprunteur selon sa tranche d'âge.

---

### 4. **Nombre de fois où l'emprunteur a été en retard de paiement de 30 à 59 jours, mais pas plus, au cours des deux dernières années**  
Ce paramètre fait référence à **l'historique des retards** de paiement de l'emprunteur, mais cette fois pour des retards plus courts (30 à 59 jours). Des retards fréquents, même s'ils ne sont pas aussi graves que les retards de 90 jours, peuvent également indiquer une **instabilité financière** ou des problèmes de gestion budgétaire. Si un emprunteur a été en retard plusieurs fois, cela pourrait signaler un **risque de non-remboursement**.

---

### 5. **Paiements mensuels des dettes, pensions alimentaires, frais de subsistance divisés par le revenu mensuel brut**  
Il s'agit du ratio des **dépenses mensuelles fixes** de l'emprunteur (dettes, pensions alimentaires, frais de subsistance) par rapport à son **revenu brut mensuel**. Ce ratio est important pour évaluer la **capacité de l'emprunteur à rembourser ses dettes**. Un ratio élevé signifie que l'emprunteur consacre une grande partie de son revenu à des paiements fixes, ce qui réduit sa **marge de manœuvre financière** et augmente le **risque de défaut**.

---

### 6. **Revenu mensuel**  
Le revenu mensuel brut est un indicateur de la **capacité financière** de l'emprunteur. Un revenu plus élevé permet généralement de mieux gérer les dettes et les dépenses courantes. Il est essentiel dans l'évaluation du **risque de crédit**, car il reflète la **solidité financière** de l'emprunteur et son aptitude à rembourser les emprunts.

---

### 7. **Nombre de prêts ouverts (à tempérament comme un prêt automobile ou un prêt hypothécaire) et de lignes de crédit (par exemple, cartes de crédit)**  
Cette variable mesure la **quantité de crédits** ouverts par l'emprunteur. Plus l'emprunteur possède de prêts ou de lignes de crédit ouvertes, plus il peut être difficile pour lui de **gérer tous ses paiements**. Un grand nombre de crédits ouverts peut être un signe d'une **gestion financière complexe** ou d'une **surendettement** potentiel. Cela peut également affecter la **notation de crédit** de l'emprunteur.

---

### 8. **Nombre de fois où l'emprunteur a été en retard de 90 jours ou plus**  
Il s'agit d'une variable qui indique combien de fois l'emprunteur a eu des **retards de paiement graves** (90 jours ou plus). Un nombre élevé de tels retards suggère que l'emprunteur a eu des **difficultés financières significatives** dans le passé, ce qui augmente fortement son **risque de crédit**.

---

### 9. **Nombre de prêts hypothécaires et immobiliers, y compris les lignes de crédit immobilier**  
Cette variable fait référence au **nombre de prêts immobiliers** ou de lignes de crédit qui sont en cours (comme un prêt hypothécaire). Les emprunteurs ayant plusieurs prêts hypothécaires peuvent avoir un **risque plus élevé**, car cela peut signaler une **dépendance à l'endettement immobilier**. Cela peut également affecter leur **capacité à payer de nouvelles dettes** en raison de la pression des paiements mensuels.

---

### 10. **Nombre de fois où l'emprunteur a été en retard de 60 à 89 jours, mais pas plus au cours des deux dernières années**  
Ce paramètre mesure l'historique des retards de paiement de l'emprunteur, mais cette fois pour des retards de **60 à 89 jours**. Bien que ces retards soient moins graves que ceux de 90 jours ou plus, ils montrent tout de même une **difficulté récurrente à payer à temps**, ce qui peut contribuer à un risque de crédit plus élevé.

---

### 11. **Nombre de personnes à charge dans la famille, à l'exclusion de l'emprunteur lui-même (conjoint, enfants, etc.)**  
Le nombre de **personnes à charge** peut affecter la capacité de l'emprunteur à rembourser ses dettes. Plus l'emprunteur a de **dépendants**, plus ses dépenses mensuelles peuvent être élevées (pour nourrir, élever, etc.), ce qui diminue sa **capacité financière** et peut augmenter son **risque de crédit**. Cette variable permet d'évaluer la **charge familiale** et l'impact qu'elle peut avoir sur la stabilité financière de l'emprunteur.

---

### 💡 **En résumé :**  
Ces variables permettent de mieux comprendre les habitudes de **paiement**, la **gestion du crédit**, et les **difficultés financières** de l'emprunteur. Elles sont cruciales dans l'évaluation du **risque de crédit** d'un individu, c'est-à-dire dans la probabilité qu'il **rembourse** ses dettes ou non. Le but du score de risque est de combiner ces facteurs pour prédire les **probabilités de défaut**.

Si tu souhaites plus de précisions sur l'une de ces variables, ou des exemples de calculs ou d'analyses, fais-le moi savoir ! 😊