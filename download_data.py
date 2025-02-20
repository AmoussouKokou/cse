import subprocess
import zipfile
import os
import shutil

# Télécharger les fichiers de la compétition Kaggle
def download_kaggle_data():
    if not os.path.exists('GiveMeSomeCredit.zip'):  # Vérifie si le fichier zip est déjà téléchargé
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", "GiveMeSomeCredit"], check=True)
            print("✅ Téléchargement terminé avec succès.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Une erreur est survenue lors du téléchargement : {e}")
    else:
        print("📂 'GiveMeSomeCredit.zip' est déjà téléchargé.")

# Décompresser le fichier GiveMeSomeCredit.zip dans le répertoire data_files
def unzip_data():
    if os.path.exists('GiveMeSomeCredit.zip'):
        if not os.path.exists('data_files'):  # Vérifie si le dossier data_files existe
            os.makedirs('data_files')  # Crée le répertoire data_files s'il n'existe pas
        if not os.path.exists('data_files/cs-training.csv'):  # Vérifie si les fichiers sont déjà extraits
            with zipfile.ZipFile('GiveMeSomeCredit.zip', 'r') as zip_ref:
                zip_ref.extractall('data_files')  # Décompresse dans data_files
            print("✅ Extraction de 'GiveMeSomeCredit.zip' terminée dans 'data_files'.")
        else:
            print("📂 Les fichiers sont déjà extraits dans 'data_files'.")
    else:
        print("❌ Le fichier 'GiveMeSomeCredit.zip' n'existe pas.")

# Supprimer uniquement les fichiers ZIP après extraction, pas les fichiers CSV
def delete_zip_files():
    zip_file_paths = ['GiveMeSomeCredit.zip']

    # Supprimer les fichiers ZIP
    for zip_file in zip_file_paths:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"✅ Le fichier {zip_file} a été supprimé.")
        else:
            print(f"📂 Le fichier {zip_file} n'existe pas ou a déjà été supprimé.")

# Fonction principale
def main():
    # Étape 1: Télécharger les données
    download_kaggle_data()

    # Étape 2: Décompresser le fichier GiveMeSomeCredit.zip dans data_files
    unzip_data()

    # Étape 5: Supprimer les fichiers ZIP et les fichiers dézippés après extraction
    delete_zip_files()

# Appeler la fonction principale
if __name__ == "__main__":
    main()
