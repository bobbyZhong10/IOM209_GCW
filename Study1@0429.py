import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import statsmodels.api as sm

# Reading data
data_path = 'panel data.csv'
data = pd.read_csv(data_path)
original_columns = data.columns.tolist()

# Prepare columns for analysis
analyst_columns = ['Numpstisten_A', 'Numneusten_A', 'Numnegasten_A', 'Numsten_A']
gb_columns = ['Tpostnum_G', 'Pospostnum_G', 'Negpostnum_G', 'Readnum_G', 'Commentnum_G']

# Ensure the NeutralWeight column exists
data['NeutralWeight'] = np.nan

# Calculate stock bar emotion score
def calculate_neutral_weight(pos_count, neg_count):
    return 0.3 if pos_count > neg_count else -0.3 if pos_count < neg_count else 0.1

if data[gb_columns].notna().all(axis=1).any():
    data.loc[data[gb_columns].notna().all(axis=1), 'NeutralWeight'] = data.apply(
        lambda x: calculate_neutral_weight(x['Pospostnum_G'], x['Negpostnum_G']), axis=1)

data['GB_emotion'] = np.nan
if 'NeutralWeight' in data.columns and 'Neupostnum_G' in data.columns:
    data['GB_emotion'] = (3 * data['Pospostnum_G'] + data['NeutralWeight'] * data['Neupostnum_G'] - 3 * data['Negpostnum_G']) / \
                         (data['Pospostnum_G'] + data['Negpostnum_G'] + data['Neupostnum_G'])

# PCA analysis
transformer = FunctionTransformer(np.log1p, validate=True)
X_transformed = transformer.fit_transform(data.loc[data[gb_columns].notna().all(axis=1), gb_columns])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
pca = PCA(n_components=2)
PCs = pca.fit_transform(X_scaled)

# Create exogenous variables
exog = pd.DataFrame(PCs, columns=['PC1', 'PC2'], index=data.loc[data[gb_columns].notna().all(axis=1)].index)
exog = sm.add_constant(exog)

# Filter data
data_with_analyst = data[data[analyst_columns].notna().all(axis=1) & data['Wretnd'].notna()]
if not data_with_analyst.empty:
    X, y = exog.loc[data_with_analyst.index], data_with_analyst['Wretnd']

    # Define the space of hyperparameters
    space = {
        'n_estimators': hp.choice('n_estimators', range(20, 201, 10)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'max_depth': hp.choice('max_depth', range(3, 14, 1)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'min_samples_split': hp.choice('min_samples_split', range(2, 11, 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 15, 1)),
        'random_state': 42
    }

    # Objective function
    def objective(params):
        gbm_model = GradientBoostingRegressor(**params)
        score = cross_val_score(gbm_model, X, y, scoring='neg_mean_squared_error', cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # Run the algorithm
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
    best_params['n_estimators'] += 20  # correct zero-based index
    best_params['max_depth'] += 3
    best_params['min_samples_split'] += 2
    best_params['min_samples_leaf'] += 1

    # Train with the best parameters
    gbm_model = GradientBoostingRegressor(**best_params)
    gbm_model.fit(X, y)
    data['Analyst_emotion_GBM'] = np.nan
    data.loc[data_with_analyst.index, 'Analyst_emotion_GBM'] = gbm_model.predict(X)

# Output to new CSV file
output_columns = original_columns + ['GB_emotion', 'Analyst_emotion_GBM']
output_data = data[output_columns]
output_data.to_csv('study1_tuned.csv', index=False)
