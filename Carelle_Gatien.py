"""
SYSTÈME DE PRÉDICTION MÉTÉOROLOGIQUE
Apprentissage en ligne (Online Learning) avec mise à jour continue

Objectif: Prédire les températures futures en s'améliorant automatiquement
à mesure que de nouvelles données sont disponibles dans l'API

Auteurs : Gatien et Carelle
"""


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

STATIONS = {
    '7024440' : 'MAGOG',
    '7024280' : 'LENNOXVILLE',
    '7020860' : 'BROMPTONVILLE',
    '7021840' : 'COATICOOK'
}

BASE_URL = "https://api.weather.gc.ca/collections/climate-daily/items"

# Fichiers de persistance du modèle
MODEL_FILE = 'model_meteo.pkl'
SCALER_FILE = 'scaler_model_meteo.pkl'
TRAINING_HISTORY_FILE = 'historique_apprentissage.json'

# Hyperparamètres
LEARNING_RATE = 0.001
WINDOW_SIZE = 7  # Utiliser les 7 derniers jours pour prédire
PREDICTION_HORIZON = 1  # Prédire 1 jour dans le futur

print("\n" + "=" * 80)
print("  SYSTÈME DE PRÉDICTION MÉTÉOROLOGIQUE - APPRENTISSAGE EN LIGNE")
print("  Sherbrooke - Prévision des températures")
print("=" * 80 + "\n")


# ==========================================
# 1. RÉCUPÉRATION DES DONNÉES
# ==========================================

def telecharger_donnees_recentes(stations_dict, start_year, end_year):
    """
    Télécharge les données météo de TOUTES les stations de l'API.

    Args:
        stations_dict (dict): Dictionnaire {climate_id: nom_station}
        start_year (int): Année de début
        end_year (int): Année de fin

    Returns:
        pd.DataFrame: Données météo combinées de toutes les stations
    """
    print(f" Téléchargement des données de {len(stations_dict)} stations...")
    print(f"   Période: {start_year} à {end_year}\n")

    all_data = []

    for climate_id, station_name in stations_dict.items():
        print(f" Station: {station_name} (ID: {climate_id})")
        station_count = 0

        for year in range(start_year, end_year + 1):
            params = {
                'f': 'json',
                'CLIMATE_IDENTIFIER': climate_id,
                'LOCAL_YEAR': year,
                'limit': 400
            }

            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])

                    for feature in features:
                        props = feature.get('properties', {})
                        # Ajouter l'identifiant de station pour distinguer les sources
                        props['STATION_ID'] = climate_id
                        props['STATION_NAME_SHORT'] = station_name
                        all_data.append(props)

                    if len(features) > 0:
                        station_count += len(features)

                import time
                time.sleep(0.05)

            except Exception as e:
                pass

        if station_count > 0:
            print(f"  ✓ Total: {station_count} enregistrements")
        else:
            print(f" Aucune donnée disponible")
        print()

    df = pd.DataFrame(all_data)
    print(f" Total combiné: {len(df)} enregistrements de {len(stations_dict)} stations\n")

    return df


def preparer_donnees(df):
    """
    Nettoie et prépare les données pour l'apprentissage.
    Gère les données de multiples stations.

    Args:
        df (pd.DataFrame): Données brutes de toutes les stations

    Returns:
        pd.DataFrame: Données nettoyées et enrichies
    """
    print(" Préparation des données de toutes les stations...")

    print(f"   Enregistrements initiaux: {len(df)}")

    # Conversion des dates
    df['LOCAL_DATE'] = pd.to_datetime(df['LOCAL_DATE'], errors='coerce')

    # Conversion des valeurs numériques
    numeric_cols = ['MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
                    'TOTAL_PRECIPITATION', 'TOTAL_RAIN', 'TOTAL_SNOW']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Supprimer les lignes avec température manquante
    df = df.dropna(subset=['MEAN_TEMPERATURE'])

    # Tri par date et station
    df = df.sort_values(['STATION_ID', 'LOCAL_DATE']).reset_index(drop=True)

    df.head(500)

    # Créer des features temporelles
    df['DAY_OF_YEAR'] = df['LOCAL_DATE'].dt.dayofyear
    df['MONTH'] = df['LOCAL_DATE'].dt.month
    df['YEAR'] = df['LOCAL_DATE'].dt.year
    df['DAY_OF_WEEK'] = df['LOCAL_DATE'].dt.dayofweek

    # Features cycliques pour capturer la saisonnalité
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_YEAR'] / 365.25)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_YEAR'] / 365.25)

    # Statistiques par station
    stations_unique = df['STATION_ID'].nunique()
    print(f"   ✓ {stations_unique} stations avec données valides")
    print(f"   ✓ {len(df)} enregistrements après nettoyage")

    # Afficher répartition par station
    print("\n Répartition par station:")
    station_counts = df.groupby('STATION_NAME_SHORT').size().sort_values(ascending=False)
    for station, count in station_counts.items():
        print(f"      • {station}: {count:,} enregistrements")

    print()

    return df


# ==========================================
# 2. CRÉATION DES FEATURES POUR ML
# ==========================================

def creer_features_temporelles(df, window_size=7):
    """
    Crée des features basées sur l'historique (fenêtre glissante).
    Traite chaque station séparément puis combine les résultats.

    Args:
        df (pd.DataFrame): Données météo de toutes les stations
        window_size (int): Taille de la fenêtre temporelle

    Returns:
        pd.DataFrame: Features pour l'apprentissage
    """
    print(f" Création des features temporelles (fenêtre: {window_size} jours)...")
    print("   Traitement par station pour préserver la continuité temporelle...\n")

    features_list = []

    # Traiter chaque station séparément
    for station_id in df['STATION_ID'].unique():
        df_station = df[df['STATION_ID'] == station_id].copy()
        df_station = df_station.sort_values('LOCAL_DATE').reset_index(drop=True)

        station_name = df_station['STATION_NAME_SHORT'].iloc[0]

        # Créer les features pour cette station
        for i in range(window_size, len(df_station)):
            # Données historiques (fenêtre)
            window = df_station.iloc[i - window_size:i]

            # Target: température du jour suivant
            target = df_station.iloc[i]['MEAN_TEMPERATURE']

            # Vérifier que le target n'est pas NaN
            if pd.isna(target):
                continue

            # Calculer les features avec gestion des NaN
            temp_mean = window['MEAN_TEMPERATURE'].mean()
            temp_std = window['MEAN_TEMPERATURE'].std()
            temp_min = window['MIN_TEMPERATURE'].min()
            temp_max = window['MAX_TEMPERATURE'].max()

            # Si trop de valeurs manquantes dans la fenêtre, skip
            if window['MEAN_TEMPERATURE'].isna().sum() > window_size * 0.3:  # Plus de 30% manquant
                continue

            # Remplacer les NaN restants par la moyenne de la fenêtre
            temp_mean = temp_mean if not pd.isna(temp_mean) else 0
            temp_std = temp_std if not pd.isna(temp_std) else 0
            temp_min = temp_min if not pd.isna(temp_min) else temp_mean
            temp_max = temp_max if not pd.isna(temp_max) else temp_mean

            # Températures lag
            temp_lag_1 = window['MEAN_TEMPERATURE'].iloc[-1]
            temp_lag_2 = window['MEAN_TEMPERATURE'].iloc[-2]
            temp_lag_3 = window['MEAN_TEMPERATURE'].iloc[-3]

            # Remplacer par la moyenne si NaN
            temp_lag_1 = temp_lag_1 if not pd.isna(temp_lag_1) else temp_mean
            temp_lag_2 = temp_lag_2 if not pd.isna(temp_lag_2) else temp_mean
            temp_lag_3 = temp_lag_3 if not pd.isna(temp_lag_3) else temp_mean

            # Tendance
            if not pd.isna(window['MEAN_TEMPERATURE'].iloc[-1]) and not pd.isna(window['MEAN_TEMPERATURE'].iloc[0]):
                temp_trend = window['MEAN_TEMPERATURE'].iloc[-1] - window['MEAN_TEMPERATURE'].iloc[0]
            else:
                temp_trend = 0

            # Précipitations avec gestion des NaN
            precip_sum = window['TOTAL_PRECIPITATION'].sum() if 'TOTAL_PRECIPITATION' in window.columns else 0
            precip_mean = window['TOTAL_PRECIPITATION'].mean() if 'TOTAL_PRECIPITATION' in window.columns else 0

            precip_sum = precip_sum if not pd.isna(precip_sum) else 0
            precip_mean = precip_mean if not pd.isna(precip_mean) else 0

            # Features statistiques sur la fenêtre
            feature_dict = {
                # Températures passées
                'temp_mean_window': temp_mean,
                'temp_std_window': temp_std,
                'temp_min_window': temp_min,
                'temp_max_window': temp_max,
                'temp_trend': temp_trend,

                # Précipitations passées
                'precip_sum_window': precip_sum,
                'precip_mean_window': precip_mean,

                # Features temporelles
                'day_of_year': df_station.iloc[i]['DAY_OF_YEAR'],
                'month': df_station.iloc[i]['MONTH'],
                'day_of_week': df_station.iloc[i]['DAY_OF_WEEK'],
                'day_sin': df_station.iloc[i]['DAY_SIN'],
                'day_cos': df_station.iloc[i]['DAY_COS'],

                # Températures des derniers jours (lag features)
                'temp_lag_1': temp_lag_1,
                'temp_lag_2': temp_lag_2,
                'temp_lag_3': temp_lag_3,

                # Identifiant de station (encoding)
                'station_id': station_id,

                # Target
                'target': target,
                'date': df_station.iloc[i]['LOCAL_DATE']
            }

            features_list.append(feature_dict)

        print(f"   ✓ {station_name}: {len([f for f in features_list if f['station_id'] == station_id])} exemples créés")

    features_df = pd.DataFrame(features_list)

    # Encoder les IDs de station (Label Encoding)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    features_df['station_encoded'] = le.fit_transform(features_df['station_id'])

    # Vérification finale : supprimer les lignes avec des NaN restants
    print(f"\n   Vérification des valeurs manquantes...")
    nan_count_before = features_df.isna().sum().sum()
    if nan_count_before > 0:
        print(f"  {nan_count_before} valeurs NaN détectées, nettoyage en cours...")
        features_df = features_df.fillna(0)  # Remplacer les NaN restants par 0
        print(f"  Toutes les valeurs NaN remplacées")

    print(f"\n Total: {len(features_df)} exemples créés de {df['STATION_ID'].nunique()} stations")
    print(f"   Données enrichies par la diversité des stations !\n")
    print(features_df)
    return features_df

# ==========================================
# 3. MODÈLE D'APPRENTISSAGE EN LIGNE
# ==========================================

class OnlineWeatherPredictor:
    """
    Modèle de prédiction météo avec apprentissage en ligne.

    Utilise SGDRegressor qui peut être mis à jour incrémentalement
    sans retrainer sur toutes les données.
    """

    def __init__(self, learning_rate=0.001):
        """
        Initialise le modèle.

        Args:
            learning_rate (float): Taux d'apprentissage
        """
        self.model = SGDRegressor(
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=1,
            warm_start=True,  # CRUCIAL pour l'apprentissage en ligne
            random_state=42
        )

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {
            'iterations': [],
            'mse': [],
            'mae': [],
            'r2': [],
            'samples_seen': 0
        }

    def fit_initial(self, X, y):
        """
        Entraînement initial sur un batch de données.

        Args:
            X (np.array): Features
            y (np.array): Target
        """
        print(" Entraînement initial du modèle...")

        # Normalisation
        X_scaled = self.scaler.fit_transform(X)

        # Entraînement initial avec plusieurs passes
        for epoch in range(100):
            self.model.partial_fit(X_scaled, y)

        self.is_fitted = True
        self.training_history['samples_seen'] = len(X)

        # Évaluation initiale
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        self.training_history['iterations'].append(0)
        self.training_history['mse'].append(mse)
        self.training_history['mae'].append(mae)
        self.training_history['r2'].append(r2)

        print(f" Entraînement initial terminé")
        print(f"   MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}\n")

    def update_online(self, X_new, y_new, iteration):
        """
        Mise à jour incrémentale avec nouvelles données.

        Args:
            X_new (np.array): Nouvelles features
            y_new (np.array): Nouvelles targets
            iteration (int): Numéro d'itération
        """
        if not self.is_fitted:
            raise Exception("Le modèle doit être initialement entraîné avant l'apprentissage en ligne")

        # Normalisation
        X_scaled = self.scaler.transform(X_new)

        # Mise à jour incrémentale
        self.model.partial_fit(X_scaled, y_new)

        self.training_history['samples_seen'] += len(X_new)

        # Évaluation
        y_pred = self.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        mae = mean_absolute_error(y_new, y_pred)
        r2 = r2_score(y_new, y_pred)

        self.training_history['iterations'].append(iteration)
        self.training_history['mse'].append(mse)
        self.training_history['mae'].append(mae)
        self.training_history['r2'].append(r2)

        return mse, mae, r2

    def predict(self, X):
        """
        Prédiction.

        Args:
            X (np.array): Features

        Returns:
            np.array: Prédictions
        """
        if not self.is_fitted:
            raise Exception("Le modèle n'est pas entraîné")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, model_path, scaler_path, history_path):
        """Sauvegarde le modèle."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def load(self, model_path, scaler_path, history_path):
        """Charge le modèle."""
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)

            self.is_fitted = True
            return True

        return False


# ==========================================
# 4. SIMULATION D'APPRENTISSAGE EN LIGNE
# ==========================================

def simulation_apprentissage_online(features_df, predictor, batch_size=30):
    """
    Simule l'apprentissage en ligne comme si les données arrivaient progressivement.

    Args:
        features_df (pd.DataFrame): Toutes les features
        predictor (OnlineWeatherPredictor): Modèle
        batch_size (int): Taille des batchs de mise à jour
    """
    print("=" * 80)
    print("SIMULATION D'APPRENTISSAGE EN LIGNE")
    print("=" * 80 + "\n")

    # Séparer features et target
    feature_cols = [col for col in features_df.columns if col not in ['target', 'date']]
    X = features_df[feature_cols].values
    y = features_df['target'].values
    dates = features_df['date'].values

    # Split: 70% entraînement initial, 30% pour apprentissage en ligne
    split_idx = int(len(X) * 0.7)

    X_initial = X[:split_idx]
    y_initial = y[:split_idx]

    X_online = X[split_idx:]
    y_online = y[split_idx:]
    dates_online = dates[split_idx:]

    print(f" Données d'entraînement initial: {len(X_initial)} exemples")
    print(f" Données pour apprentissage en ligne: {len(X_online)} exemples")
    print(f" Taille des batchs: {batch_size} exemples\n")

    # Entraînement initial
    predictor.fit_initial(X_initial, y_initial)

    # Apprentissage en ligne par batchs
    print(" Début de l'apprentissage en ligne...\n")

    num_batches = len(X_online) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_online[start_idx:end_idx]
        y_batch = y_online[start_idx:end_idx]
        date_batch = dates_online[start_idx]

        # Mise à jour du modèle
        mse, mae, r2 = predictor.update_online(X_batch, y_batch, i + 1)

        print(f"Batch {i + 1}/{num_batches} | Date: {pd.to_datetime(date_batch).date()} | "
              f"MSE: {mse:.4f} | MAE: {mae:.4f}°C | R²: {r2:.4f}")

    print(f"\n Apprentissage en ligne terminé")
    print(f"   Total d'exemples traités: {predictor.training_history['samples_seen']}")

    # Évaluation finale
    print("\n" + "=" * 80)
    print("ÉVALUATION FINALE DU MODÈLE")
    print("=" * 80 + "\n")

    y_pred_final = predictor.predict(X)
    mse_final = mean_squared_error(y, y_pred_final)
    mae_final = mean_absolute_error(y, y_pred_final)
    r2_final = r2_score(y, y_pred_final)

    print(f" Performance sur l'ensemble des données:")
    print(f"   MSE: {mse_final:.4f}")
    print(f"   MAE: {mae_final:.4f}°C")
    print(f"   R²: {r2_final:.4f}")
    print(f"   RMSE: {np.sqrt(mse_final):.4f}°C\n")

    return X, y, y_pred_final, dates


# ==========================================
# 5. VISUALISATIONS
# ==========================================

def generer_cycle_saisonnier(df_clean):
    """
    Génère le graphique du cycle saisonnier moyen.

    Args:
        df_clean (pd.DataFrame): DataFrame avec colonnes LOCAL_DATE et MEAN_TEMPERATURE
    """
    print("=" * 80)
    print("GÉNÉRATION DU CYCLE SAISONNIER MOYEN")
    print("=" * 80 + "\n")

    # Préparation des données
    df_seasonal = df_clean.copy()
    df_seasonal['DAY_OF_YEAR'] = df_seasonal['LOCAL_DATE'].dt.dayofyear

    # Grouper par jour de l'année et calculer statistiques
    print(" Calcul des statistiques par jour de l'année...")
    seasonal_stats = df_seasonal.groupby('DAY_OF_YEAR')['MEAN_TEMPERATURE'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()

    print(f"   ✓ {len(seasonal_stats)} jours analysés")
    print(f"   ✓ Température moyenne annuelle: {seasonal_stats['mean'].mean():.2f}°C")
    print(f"   ✓ Température min observée: {seasonal_stats['min'].min():.2f}°C")
    print(f"   ✓ Température max observée: {seasonal_stats['max'].max():.2f}°C\n")

    # Création du graphique
    fig, ax = plt.subplots(figsize=(16, 7))

    # Tracer la température moyenne avec bande de confiance
    ax.plot(seasonal_stats['DAY_OF_YEAR'], seasonal_stats['mean'],
            color='#e74c3c', linewidth=3, label='Température moyenne', zorder=5)

    # Bande de confiance (± 1 écart-type)
    ax.fill_between(seasonal_stats['DAY_OF_YEAR'],
                    seasonal_stats['mean'] - seasonal_stats['std'],
                    seasonal_stats['mean'] + seasonal_stats['std'],
                    alpha=0.3, color='#e74c3c', label='± 1 écart-type', zorder=3)

    # Températures min/max observées
    ax.plot(seasonal_stats['DAY_OF_YEAR'], seasonal_stats['max'],
            color='#f39c12', linewidth=1.5, linestyle='--', alpha=0.7,
            label='Max observé', zorder=4)
    ax.plot(seasonal_stats['DAY_OF_YEAR'], seasonal_stats['min'],
            color='#3498db', linewidth=1.5, linestyle='--', alpha=0.7,
            label='Min observé', zorder=4)

    # Ligne de référence pour 0°C
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=2, alpha=0.5, zorder=2)
    ax.text(5, 1, '0°C', fontsize=10, color='gray', fontweight='bold')

    # Annotations des saisons avec zones colorées
    seasons = [
        (1, 79, 'Hiver', '#3498db'),
        (80, 171, 'Printemps', '#2ecc71'),
        (172, 265, 'Été', '#f39c12'),
        (266, 355, 'Automne', '#e67e22'),
        (356, 365, 'Hiver', '#3498db')
    ]

    y_max = seasonal_stats['max'].max()
    for start, end, season, color in seasons:
        mid = (start + end) / 2
        ax.axvspan(start, end, alpha=0.05, color=color, zorder=1)
        ax.text(mid, y_max * 0.95, season,
                ha='center', fontsize=12, fontweight='bold', color=color, zorder=6)

    # Labels et titre
    ax.set_xlabel('Jour de l\'année', fontsize=12, fontweight='bold')
    ax.set_ylabel('Température (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Cycle Saisonnier Moyen - Sherbrooke (toutes années et stations confondues)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(1, 365)

    # Ajouter les mois en x-axis
    months_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
    months_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                     'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc', '']
    ax.set_xticks(months_days)
    ax.set_xticklabels(months_labels)

    # Ajouter une grille verticale pour les mois
    for day in months_days:
        ax.axvline(x=day, color='gray', linestyle=':', alpha=0.2, zorder=0)

    plt.tight_layout()
    plt.savefig('ml_cycle_saisonnier.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: ml_cycle_saisonnier.png\n")
    plt.show()
    plt.close()

    # Statistiques détaillées par saison
    print("=" * 80)
    print("STATISTIQUES PAR SAISON")
    print("=" * 80 + "\n")

    season_ranges = {
        'Hiver (Jan-Mar)': (1, 79),
        'Printemps (Mar-Jun)': (80, 171),
        'Été (Jun-Sep)': (172, 265),
        'Automne (Sep-Déc)': (266, 355),
        'Hiver (Déc)': (356, 365)
    }

    for season_name, (start, end) in season_ranges.items():
        season_data = seasonal_stats[
            (seasonal_stats['DAY_OF_YEAR'] >= start) &
            (seasonal_stats['DAY_OF_YEAR'] <= end)
            ]

        print(f" {season_name}:")
        print(f"   • Température moyenne: {season_data['mean'].mean():.2f}°C")
        print(f"   • Température min: {season_data['min'].min():.2f}°C")
        print(f"   • Température max: {season_data['max'].max():.2f}°C")
        print(f"   • Variabilité (écart-type moyen): {season_data['std'].mean():.2f}°C")
        print()

    print("=" * 80)

def visualiser_resultats(predictor, X, y, y_pred, dates):
    """
    Crée des visualisations des résultats.

    Args:
        predictor (OnlineWeatherPredictor): Modèle entraîné
        X (np.array): Features
        y (np.array): Valeurs réelles
        y_pred (np.array): Prédictions
        dates (np.array): Dates
    """
    print("=" * 80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 80 + "\n")

    # Figure 1: Évolution de l'apprentissage
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))

    history = predictor.training_history

    # MSE
    axes1[0, 0].plot(history['iterations'], history['mse'], 'o-', color='#e74c3c', linewidth=2)
    axes1[0, 0].set_xlabel('Itération (Batch)', fontweight='bold')
    axes1[0, 0].set_ylabel('MSE', fontweight='bold')
    axes1[0, 0].set_title('Évolution du MSE (apprentissage en ligne)', fontweight='bold')
    axes1[0, 0].grid(True, alpha=0.3)

    # MAE
    axes1[0, 1].plot(history['iterations'], history['mae'], 'o-', color='#3498db', linewidth=2)
    axes1[0, 1].set_xlabel('Itération (Batch)', fontweight='bold')
    axes1[0, 1].set_ylabel('MAE (°C)', fontweight='bold')
    axes1[0, 1].set_title('Évolution du MAE (apprentissage en ligne)', fontweight='bold')
    axes1[0, 1].grid(True, alpha=0.3)

    # R²
    axes1[1, 0].plot(history['iterations'], history['r2'], 'o-', color='#2ecc71', linewidth=2)
    axes1[1, 0].set_xlabel('Itération (Batch)', fontweight='bold')
    axes1[1, 0].set_ylabel('R²', fontweight='bold')
    axes1[1, 0].set_title('Évolution du R² (apprentissage en ligne)', fontweight='bold')
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Nombre d'exemples
    axes1[1, 1].text(0.5, 0.5, f"Exemples traités:\n{history['samples_seen']:,}",
                     ha='center', va='center', fontsize=24, fontweight='bold')
    axes1[1, 1].axis('off')

    plt.suptitle('Apprentissage en Ligne - Évolution des Métriques',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ml_evolution_apprentissage.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: ml_evolution_apprentissage.png")
    plt.show()
    plt.close()

    # Figure 2: Prédictions vs Réalité (derniers 365 jours)
    fig2, ax2 = plt.subplots(figsize=(16, 6))

    # Prendre les 365 derniers jours
    last_365 = -365
    dates_plot = pd.to_datetime(dates[last_365:])
    y_plot = y[last_365:]
    y_pred_plot = y_pred[last_365:]

    ax2.plot(dates_plot, y_plot, label='Température réelle',
             color='#2c3e50', linewidth=2, alpha=0.7)
    ax2.plot(dates_plot, y_pred_plot, label='Prédiction du modèle',
             color='#e74c3c', linewidth=2, linestyle='--', alpha=0.8)

    ax2.fill_between(dates_plot, y_plot, y_pred_plot, alpha=0.2, color='gray')

    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Température (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Prédictions vs Températures Réelles (dernière année)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('ml_predictions_vs_realite.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: ml_predictions_vs_realite.png")
    plt.show()
    plt.close()

    # Figure 3: Scatter plot prédictions vs réalité
    fig3, ax3 = plt.subplots(figsize=(10, 10))

    ax3.scatter(y, y_pred, alpha=0.4, s=10, color='#3498db')

    # Ligne de référence parfaite
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')

    ax3.set_xlabel('Température réelle (°C)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Température prédite (°C)', fontsize=12, fontweight='bold')
    ax3.set_title('Corrélation Prédictions vs Réalité', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Ajouter R²
    r2 = r2_score(y, y_pred)
    ax3.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax3.transAxes,
             fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('ml_correlation.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: ml_correlation.png")
    plt.show()
    plt.close()

    # Figure 4: Distribution des erreurs
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    errors = y - y_pred

    # Histogramme des erreurs
    axes4[0].hist(errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes4[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes4[0].set_xlabel('Erreur de prédiction (°C)', fontsize=11, fontweight='bold')
    axes4[0].set_ylabel('Fréquence', fontsize=11, fontweight='bold')
    axes4[0].set_title('Distribution des erreurs', fontsize=12, fontweight='bold')
    axes4[0].grid(True, alpha=0.3, axis='y')

    # Boxplot des erreurs absolues
    axes4[1].boxplot(np.abs(errors), vert=True)
    axes4[1].set_ylabel('Erreur absolue (°C)', fontsize=11, fontweight='bold')
    axes4[1].set_title('Distribution des erreurs absolues', fontsize=12, fontweight='bold')
    axes4[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Analyse des Erreurs de Prédiction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml_erreurs.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: ml_erreurs.png")
    plt.show()
    plt.close()

    print()


# ==========================================
# 6. FONCTION PRINCIPALE
# ==========================================

def main():
    """Fonction principale du système de prédiction."""

    # 1. Télécharger les données de TOUTES les stations
    df_raw = telecharger_donnees_recentes(
        stations_dict=STATIONS,
        start_year=2000,  # 25 ans de données
        end_year=2025
    )

    # 2. Préparer les données
    df_clean = preparer_donnees(df_raw)

    # 3. Créer les features
    features_df = creer_features_temporelles(df_clean, window_size=WINDOW_SIZE)

    # 4. Initialiser ou charger le modèle
    predictor = OnlineWeatherPredictor(learning_rate=LEARNING_RATE)

    model_exists = predictor.load(MODEL_FILE, SCALER_FILE, TRAINING_HISTORY_FILE)

    if model_exists:
        print(" Modèle existant chargé\n")
    else:
        print(" Création d'un nouveau modèle\n")

    # 5. Simulation d'apprentissage en ligne
    X, y, y_pred, dates = simulation_apprentissage_online(
        features_df,
        predictor,
        batch_size=30
    )

    # 6. Sauvegarder le modèle
    predictor.save(MODEL_FILE, SCALER_FILE, TRAINING_HISTORY_FILE)
    print(f" Modèle sauvegardé: {MODEL_FILE}\n")

    # 7. Visualiser les résultats
    visualiser_resultats(predictor, X, y, y_pred, dates)
    generer_cycle_saisonnier(df_clean)

    # 8. Rapport final
    print("=" * 80)
    print("RAPPORT FINAL")
    print("=" * 80 + "\n")

    print(" CONFIGURATION:")
    print(f"   • Stations: {len(STATIONS)} stations de Sherbrooke")
    for station_id, station_name in STATIONS.items():
        print(f"      - {station_name} ({station_id})")
    print(f"   • Fenêtre temporelle: {WINDOW_SIZE} jours")
    print(f"   • Taux d'apprentissage: {LEARNING_RATE}")
    print(f"   • Horizon de prédiction: {PREDICTION_HORIZON} jour\n")

    print(" APPRENTISSAGE:")
    print(f"   • Exemples traités: {predictor.training_history['samples_seen']:,}")
    print(f"   • Nombre de mises à jour: {len(predictor.training_history['iterations'])}")
    print(f"   • Sources de données: {len(STATIONS)} stations combinées\n")

    print(" PERFORMANCE:")
    mse_final = mean_squared_error(y, y_pred)
    mae_final = mean_absolute_error(y, y_pred)
    r2_final = r2_score(y, y_pred)
    print(f"   • MAE: {mae_final:.4f}°C")
    print(f"   • RMSE: {np.sqrt(mse_final):.4f}°C")
    print(f"   • R²: {r2_final:.4f}\n")

    print(" INTERPRÉTATION:")
    print(f"   Le modèle prédit la température avec une erreur moyenne de {mae_final:.2f}°C.")
    print(f"   Le R² de {r2_final:.3f} indique que le modèle explique")
    print(f"   {r2_final * 100:.1f}% de la variance des températures.")
    print(f"\n    Avantage multi-stations: Le modèle capture la variabilité spatiale")
    print(f"   et temporelle de Sherbrooke, le rendant plus robuste et généralisable!\n")

    print("=" * 80)
    print(" SYSTÈME DE PRÉDICTION PRÊT")
    print("=" * 80)
    print("\nLe modèle continuera à s'améliorer avec chaque nouvelle mise à jour de données")
    print("de TOUTES les 6 stations de Sherbrooke!\n")


# ==========================================
# EXÉCUTION
# ==========================================

if __name__ == "__main__":
    main()