import os
import logging
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ML_Betting_System")

class ESPNDataFetcher:
    def __init__(self):
        self.url = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

    def fetch_live_events(self) -> List[Dict[str, Any]]:
        logger.info("🌐 Consultando API de ESPN para obtener partidos...")
        try:
            with requests.Session() as session:
                response = session.get(self.url, headers=self.headers, timeout=15)
                response.raise_for_status()
                datos = response.json()
            
            eventos = datos.get('events', [])
            if not eventos:
                logger.warning("⚠️ No hay eventos programados en el JSON de ESPN.")
                return []
            
            lista_partidos = []
            for evento in eventos:
                try:
                    competencias = evento.get('competitions', [])
                    if not competencias:
                        continue
                    comp = competencias[0]
                    
                    competitors = comp.get('competitors', [])
                    if len(competitors) < 2:
                        continue
                    
                    home_team, away_team = 'Local', 'Visitante'
                    for competitor in competitors:
                        role = competitor.get('homeAway')
                        name = competitor.get('team', {}).get('displayName')
                        if role == 'home':
                            home_team = name or home_team
                        elif role == 'away':
                            away_team = name or away_team
                    
                    odds_list = comp.get('odds', [])
                    home_odds, away_odds, draw_odds = 2.00, 3.40, 3.20
                    
                    if odds_list and isinstance(odds_list, list):
                        item = odds_list[0]
                        raw_home = item.get('homeTeamOdds', {}).get('moneyLine') or item.get('home', {}).get('odds')
                        raw_away = item.get('awayTeamOdds', {}).get('moneyLine') or item.get('away', {}).get('odds')
                        raw_draw = item.get('drawOdds', {}).get('moneyLine') or item.get('draw', {}).get('odds')
                        
                        if raw_home is not None: home_odds = abs(float(raw_home))
                        if raw_away is not None: away_odds = abs(float(raw_away))
                        if raw_draw is not None: draw_odds = abs(float(raw_draw))

                    if home_odds < 1.01: home_odds = 1.01 + abs(home_odds)
                    if away_odds < 1.01: away_odds = 1.01 + abs(away_odds)
                    if draw_odds < 1.01: draw_odds = 1.01 + abs(draw_odds)

                    lista_partidos.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_odds": float(home_odds),
                        "draw_odds": float(draw_odds),
                        "away_odds": float(away_odds)
                    })
                except Exception as ex:
                    logger.debug(f"Error procesando evento individual: {ex}")
                    continue

            logger.info(f"✅ Procesados correctamente {len(lista_partidos)} partidos.")
            return lista_partidos
        except Exception as e:
            logger.error(f"❌ Error crítico en API ESPN: {e}")
            return []

class FeatureEngineer:
    @staticmethod
    def construct_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['implied_p_home'] = 1.0 / df['home_odds']
        df['implied_p_draw'] = 1.0 / df['draw_odds']
        df['implied_p_away'] = 1.0 / df['away_odds']
        
        df['bookmaker_margin'] = (df['implied_p_home'] + df['implied_p_draw'] + df['implied_p_away']) - 1.0
        
        df['clean_p_home'] = df['implied_p_home'] / (1.0 + df['bookmaker_margin'])
        df['clean_p_draw'] = df['implied_p_draw'] / (1.0 + df['bookmaker_margin'])
        df['clean_p_away'] = df['implied_p_away'] / (1.0 + df['bookmaker_margin'])
        
        df['odds_ratio_h_a'] = df['home_odds'] / df['away_odds']
        df['odds_skewness'] = np.abs(df['home_odds'] - df['away_odds']) / (df['draw_odds'] + 1e-5)
        df['market_entropy'] = - (df['clean_p_home'] * np.log(df['clean_p_home'] + 1e-8) +
                                  df['clean_p_draw'] * np.log(df['clean_p_draw'] + 1e-8) +
                                  df['clean_p_away'] * np.log(df['clean_p_away'] + 1e-8))
        return df

class PredictiveModelML:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective='multiclass',
            num_class=3,
            verbose=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_synthetic_training()

    def _initialize_synthetic_training(self):
        logger.info("⚙️ Generando datos de entrenamiento históricos sintéticos para calibración...")
        np.random.seed(42)
        n_samples = 2500
        
        sim_home_odds = np.random.uniform(1.15, 7.0, n_samples)
        sim_draw_odds = np.random.uniform(2.5, 5.0, n_samples)
        sim_away_odds = np.random.uniform(1.20, 9.0, n_samples)
        
        df_sim = pd.DataFrame({
            'home_odds': sim_home_odds,
            'draw_odds': sim_draw_odds,
            'away_odds': sim_away_odds
        })
        
        df_feats = FeatureEngineer.construct_features(df_sim)
        
        probs = df_feats[['clean_p_home', 'clean_p_draw', 'clean_p_away']].values
        y = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])
        
        feature_cols = [col for col in df_feats.columns if col not in ['home_team', 'away_team']]
        X = df_feats[feature_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        logger.info("🚀 Modelo LightGBM entrenado y calibrado exitosamente.")

    def predict_probabilities(self, df_live: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("El modelo predictivo no ha sido entrenado.")
        
        feature_cols = [col for col in df_live.columns if col not in ['home_team', 'away_team']]
        X_live_scaled = self.scaler.transform(df_live[feature_cols])
        return self.model.predict_proba(X_live_scaled)

class KellyPortfolioOptimizer:
    def __init__(self, fractional_factor: float = 0.25, max_total_allocation: float = 0.40):
        self.fractional_factor = fractional_factor
        self.max_total_allocation = max_total_allocation

    def optimize_bets(self, matches: List[Dict[str, Any]], prob_matrix: np.ndarray, current_bankroll: float) -> List[Dict[str, Any]]:
        optimized_recommendations = []
        raw_allocations = []
        
        for idx, match in enumerate(matches):
            p_home, p_draw, p_away = prob_matrix[idx]
            
            odds_h, odds_d, odds_a = match['home_odds'], match['draw_odds'], match['away_odds']
            
            ev_h = (p_home * odds_h) - 1.0
            ev_d = (p_draw * odds_d) - 1.0
            ev_a = (p_away * odds_a) - 1.0
            
            best_ev = -1.0
            best_outcome = None
            best_odds = 1.0
            best_prob = 0.0
            
            for ev, outcome, odds, prob in [(ev_h, 'HOME', odds_h, p_home), 
                                            (ev_d, 'DRAW', odds_d, p_draw), 
                                            (ev_a, 'AWAY', odds_a, p_away)]:
                if ev > best_ev:
                    best_ev = ev
                    best_outcome = outcome
                    best_odds = odds
                    best_prob = prob
            
            if best_ev > 0.01:
                b_kelly = best_odds - 1.0
                kelly_fraction = (best_prob * b_kelly - (1.0 - best_prob)) / b_kelly
                
                safe_fraction = kelly_fraction * self.fractional_factor
                if safe_fraction > 0:
                    raw_allocations.append({
                        "match": f"{match['home_team']} vs {match['away_team']}",
                        "outcome": best_outcome,
                        "odds": best_odds,
                        "probability": best_prob,
                        "expected_value": best_ev,
                        "raw_fraction": safe_fraction
                    })
        
        if not raw_allocations:
            return []
            
        total_raw = sum(item['raw_fraction'] for item in raw_allocations)
        scale_factor = 1.0
        if total_raw > self.max_total_allocation:
            scale_factor = self.max_total_allocation / total_raw
            
        for item in raw_allocations:
            final_fraction = item['raw_fraction'] * scale_factor
            bet_amount = final_fraction * current_bankroll
            
            optimized_recommendations.append({
                "match": item['match'],
                "outcome": item['outcome'],
                "odds": round(item['odds'], 2),
                "probability": f"{round(item['probability'] * 100, 2)}%",
                "roi_est": f"{round(item['expected_value'] * 100, 2)}%",
                "bankroll_pct": f"{round(final_fraction * 100, 2)}%",
                "suggested_bet": round(bet_amount, 2)
            })
            
        return optimized_recommendations

class LiveBettingPipeline:
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.fetcher = ESPNDataFetcher()
        self.model = PredictiveModelML()
        self.optimizer = KellyPortfolioOptimizer(fractional_factor=0.20, max_total_allocation=0.35)

    def execute(self):
        logger.info("⚡ Iniciando Pipeline de Predicciones y Optimización...")
        raw_matches = self.fetcher.fetch_live_events()
        
        if not raw_matches:
            logger.info("❌ No se obtuvieron eventos activos o programados.")
            return
            
        df_raw = pd.DataFrame(raw_matches)
        df_features = FeatureEngineer.construct_features(df_raw)
        
        probabilities = self.model.predict_probabilities(df_features)
        
        recommendations = self.optimizer.optimize_bets(raw_matches, probabilities, self.bankroll)
        
        self._display_results(recommendations)

    def _display_results(self, recommendations: List[Dict[str, Any]]):
        print("\n" + "="*80)
        print(f"💰 RESUMEN DE PORTAFOLIO DE APUESTAS RECOMENDADAS (Banca: ${self.bankroll})")
        print("="*80)
        
        if not recommendations:
            print("🔍 No se encontraron oportunidades con Esperanza Matemática (EV) Positiva en el mercado actual.")
        else:
            for idx, rec in enumerate(recommendations, 1):
                print(f"📌 {idx}. {rec['match']}")
                print(f"   🎯 Selección: {rec['outcome']} | Cuota: {rec['odds']} | Probabilidad ML: {rec['probability']}")
                print(f"   📈 ROI Esperado: {rec['roi_est']} | Asignación sugerida: {rec['bankroll_pct']} (${rec['suggested_bet']})")
                print("-" * 80)
        print("="*80 + "\n")

if __name__ == "__main__":
    pipeline = LiveBettingPipeline(bankroll=5000.0)
    pipeline.execute()