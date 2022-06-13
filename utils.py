import time
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def fit_model(model, X_train, y_train, X_test, param_grid, 
              cv=5, scoring_fit='neg_root_mean_squared_error'):
    if param_grid:
        gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2)

        ti = time.time()
        fitted_model = gs.fit(X_train, y_train)
        fit_time = time.time() - ti
        pred = fitted_model.predict(X_test)
    else:
        ti = time.time() 
        fitted_model = model.fit(X_train, y_train)
        fit_time = time.time() - ti
        pred = fitted_model.predict(X_test)
        
    return fitted_model, f"{fit_time:.2f}", pred


def evaluate_model(model, X, y_act):
    y_pred = model.predict(X)
    r2 = r2_score(y_act, y_pred)
    mae = mean_absolute_error(y_act, y_pred)
    rmse_val = mean_squared_error(y_act, y_pred, squared=False)
    return r2, mae, rmse_val

def fit_evaluate_model(model, X_train, y_train, X_val, y_act_val, feature, param_grid = None):
    model, fit_time, pred = fit_model(model, X_train, y_train, X_val, param_grid)
    r2_train, mae_train, rmse_train = evaluate_model(model, X_train, y_train)
    r2_val, mae_val, rmse_val = evaluate_model(model, X_val, y_act_val)
    result_dict = {
        'feature': feature,
        'model_name': type(model).__name__,
        'model': model,
        'fit_time': fit_time,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val,
        'y_test': y_act_val,
        'predict': pred,
        }
    return model, result_dict

def append_result_df(df, result_dict):
    df_result_appended = df.append(result_dict, ignore_index=True)
    return df_result_appended

def append_model_dict(dic, model, feature):
    dic[feature] = model
    return dic

def corr_reduction(corr, thresh, data):    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= thresh:
                if columns[j]:
                    columns[j] = False
    selected_columns = data.columns[columns]
    data = data[selected_columns]
    return data