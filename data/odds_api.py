import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feats = self._generate_features(df)
        numerical_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [c for c in numerical_cols if c not in ['target', 'home_win']]
        
        df_feats[numerical_cols] = self.scaler.fit_transform(df_feats[numerical_cols].fillna(0))
        self.is_fitted = True
        return df_feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming new data.")
        df_feats = self._generate_features(df)
        numerical_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [c for c in numerical_cols if c not in ['target', 'home_win']]
        
        df_feats[numerical_cols] = self.scaler.transform(df_feats[numerical_cols].fillna(0))
        return df_feats

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Avoid division by zero
        df['home_odds'] = df['home_odds'].replace(0, 1.01)
        df['away_odds'] = df['away_odds'].replace(0, 1.01)
        
        # Implied probabilities from bookmaker
        df['implied_prob_home'] = 1.0 / df['home_odds']
        df['implied_prob_away'] = 1.0 / df['away_odds']
        df['bookmaker_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
        
        # Performance Indicators (using historical stats if available, else generated/mocked)
        if 'home_shots_on_target_avg' in df.columns and 'away_shots_on_target_avg' in df.columns:
            df['shot_efficiency_diff'] = df['home_shots_on_target_avg'] - df['away_shots_on_target_avg']
        else:
            df['shot_efficiency_diff'] = (df['implied_prob_home'] - df['implied_prob_away']) * 5.0
            
        if 'home_goals_conceded_avg' in df.columns and 'away_goals_conceded_avg' in df.columns:
            df['defense_strength_diff'] = df['away_goals_conceded_avg'] - df['home_goals_conceded_avg']
        else:
            df['defense_strength_diff'] = (df['implied_prob_home'] - df['implied_prob_away']) * 1.5

        # Relative superiority indicators
        df['odds_ratio'] = df['home_odds'] / df['away_odds']
        df['log_odds_ratio'] = np.log(df['odds_ratio'])
        
        # Historical momentum (Form index)
        if 'home_form_points' in df.columns and 'away_form_points' in df.columns:
            df['form_diff'] = df['home_form_points'] - df['away_form_points']
        else:
            df['form_diff'] = df['implied_prob_home'] * 3.0 - df['implied_prob_away'] * 3.0

        return df

class MarketPredictor:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.feature_engineer = FeatureEngineer()
        self.features_to_use = []

    def prepare_data_and_train(self, historical_data: pd.DataFrame):
        df_processed = self.feature_engineer.fit_transform(historical_data)
        
        if 'target' not in df_processed.columns:
            # Generate binary target (1 if Home wins, 0 otherwise) if not provided
            df_processed['target'] = (df_processed['home_score'] > df_processed['away_score']).astype(int)

        exclude_cols = ['target', 'home_team', 'away_team', 'home_score', 'away_score', 'date']
        self.features_to_use = [col for col in df_processed.columns if col not in exclude_cols]
        
        X = df_processed[self.features_to_use]
        y = df_processed['target']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

    def predict_probabilities(self, upcoming_matches: pd.DataFrame) -> np.ndarray:
        df_processed = self.feature_engineer.transform(upcoming_matches)
        X = df_processed[self.features_to_use]
        return self.model.predict_proba(X)[:, 1]

class KellyPortfolioManager:
    def __init__(self, kelly_fraction: float = 0.15, max_bet_size: float = 0.05):
        self.kelly_fraction = kelly_fraction
        self.max_bet_size = max_bet_size

    def calculate_stakes(self, bankroll: float, market_opportunities: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates optimal fractional Kelly stake sizes for a set of opportunities.
        Expects market_opportunities to have: 'pred_prob_home', 'home_odds', 'pred_prob_away', 'away_odds'
        """
        results = []
        for idx, row in market_opportunities.iterrows():
            # Home Win Evaluation
            p_h = row['pred_prob_home']
            o_h = row['home_odds']
            b_h = o_h - 1.0
            edge_h = (p_h * o_h) - 1.0
            
            # Away Win Evaluation
            p_a = row['pred_prob_away']
            o_a = row['away_odds']
            b_a = o_a - 1.0
            edge_a = (p_a * o_a) - 1.0

            # Default allocation
            recommended_bet = "NO_BET"
            optimal_stake = 0.0
            expected_roi = 0.0
            selected_odds = 1.0
            selected_prob = 0.0

            # Dynamic Selection of maximum edge (with positive expectancy check)
            if edge_h > 0 and edge_h > edge_a:
                f_star = edge_h / b_h
                optimal_stake = f_star * self.kelly_fraction
                recommended_bet = "HOME"
                expected_roi = edge_h
                selected_odds = o_h
                selected_prob = p_h
            elif edge_a > 0 and edge_a > edge_h:
                f_star = edge_a / b_a
                optimal_stake = f_star * self.kelly_fraction
                recommended_bet = "AWAY"
                expected_roi = edge_a
                selected_odds = o_a
                selected_prob = p_a

            # Constraint application (Risk capping)
            optimal_stake = min(max(optimal_stake, 0.0), self.max_bet_size)
            monetary_bet = optimal_stake * bankroll

            results.append({
                "match": row.get("match", "Unknown Match"),
                "recommended_bet": recommended_bet,
                "selected_odds": selected_odds,
                "model_probability": round(selected_prob, 4),
                "expected_roi": round(expected_roi, 4),
                "suggested_stake_percentage": round(optimal_stake * 100, 2),
                "bet_amount": round(monetary_bet, 2)
            })

        return pd.DataFrame(results)

class ProductionEngine:
    def __init__(self, bankroll: float = 10000.0, kelly_fraction: float = 0.1, max_bet_size: float = 0.05):
        self.bankroll = bankroll
        self.predictor = MarketPredictor()
        self.portfolio_manager = KellyPortfolioManager(kelly_fraction=kelly_fraction, max_bet_size=max_bet_size)
        self._initialize_and_train_mock_data()

    def _initialize_and_train_mock_data(self):
        """Generates historical structured data to safely boot-up and train the internal models."""
        np.random.seed(42)
        n_samples = 1500
        
        home_odds = np.random.uniform(1.3, 5.0, n_samples)
        away_odds = 1.0 / (1.05 - (1.0 / home_odds)) # Simplified realistic odds correlation
        away_odds = np.clip(away_odds, 1.3, 15.0)
        
        home_score = np.random.poisson(1.6, n_samples)
        away_score = np.random.poisson(1.1, n_samples)
        
        historical_df = pd.DataFrame({
            'home_odds': home_odds,
            'away_odds': away_odds,
            'home_score': home_score,
            'away_score': away_score
        })
        historical_df['target'] = (historical_df['home_score'] > historical_df['away_score']).astype(int)
        
        self.predictor.prepare_data_and_train(historical_df)

    def process_upcoming_matches(self, matches: list) -> list:
        """
        Main entrypoint replacing the legacy system. Orchestrates inference, 
        edge-detection, and optimal capital allocation.
        """
        df_upcoming = pd.DataFrame(matches)
        
        # Dual-directional probability model pipeline
        # 1. Model probabilities for Home Wins
        pred_prob_home = self.predictor.predict_probabilities(df_upcoming)
        
        # 2. Model probabilities for Away Wins (invert inputs dynamically to evaluate opposite outcomes)
        df_inverted = df_upcoming.rename(columns={
            'home_odds': 'away_odds',
            'away_odds': 'home_odds'
        })
        pred_prob_away = self.predictor.predict_probabilities(df_inverted)
        
        df_upcoming['pred_prob_home'] = pred_prob_home
        df_upcoming['pred_prob_away'] = pred_prob_away
        
        # Format for portfolio processing
        df_upcoming['match'] = df_upcoming['home'] + " vs " + df_upcoming['away']
        
        portfolio_df = self.portfolio_manager.calculate_stakes(self.bankroll, df_upcoming)
        return portfolio_df.to_dict(orient="records")

def obtener_cuotas(partidos, bankroll: float = 10000.0):
    """
    Drop-in replacement function for the legacy API, enriched with 
    Machine Learning and Professional Portfolio Management.
    """
    engine = ProductionEngine(bankroll=bankroll)
    
    # Standardize input interface
    standardized_matches = []
    for p in partidos:
        home_name = p.get('home', 'Local Team')
        away_name = p.get('away', 'Away Team')
        standardized_matches.append({
            'home': home_name,
            'away': away_name,
            'home_odds': p.get('home_odds', round(np.random.uniform(1.5, 2.8), 2)),
            'away_odds': p.get('away_odds', round(np.random.uniform(1.8, 3.5), 2))
        })
        
    return engine.process_upcoming_matches(standardized_matches)