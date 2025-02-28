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

# Décompresser le fichier GiveMeSomeCredit.zip dans le répertoire data/raw
def unzip_data():
    if os.path.exists('GiveMeSomeCredit.zip'):
        if not os.path.exists('data/raw'):  # Vérifie si le dossier data/raw existe
            os.makedirs('data/raw')  # Crée le répertoire data/raw s'il n'existe pas
        if not os.path.exists('data/raw/cs-training.csv'):  # Vérifie si les fichiers sont déjà extraits
            with zipfile.ZipFile('GiveMeSomeCredit.zip', 'r') as zip_ref:
                zip_ref.extractall('data/raw')  # Décompresse dans data/raw
            print("✅ Extraction de 'GiveMeSomeCredit.zip' terminée dans 'data/raw'.")
        else:
            print("📂 Les fichiers sont déjà extraits dans 'data/raw'.")
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

def create_folders():
    # Créer les dossiers
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists('data/external'):
        os.makedirs('data/external')
    
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    
    if not os.path.exists('config'):
        os.makedirs('config')

    if not os.path.exists('notebooks'):
        os.makedirs('notebooks')
    
    if not os.path.exists('src'):
        os.makedirs('src')

# Fonction principale
def main():
    # Créer les dossiers
    create_folders()

    # Étape 1: Télécharger les données
    download_kaggle_data()

    # Étape 2: Décompresser le fichier GiveMeSomeCredit.zip dans data/raw
    unzip_data()

    # Étape 5: Supprimer les fichiers ZIP et les fichiers dézippés après extraction
    delete_zip_files()

    # Créer le répertoire data/processed si il n'existe pas
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

# Appeler la fonction principale
if __name__ == "__main__":
    main()
