import numpy as np


def predict_model(model, X_seq, scaler_y):
    # Executa a inferência do modelo sobre as sequências 
    preds_scaled = model.predict(X_seq, verbose=0)

    # Inverte a normalização para retornar aos valores originais de venda
    # O reshape adapta o vetor para o formato esperado pelo scikit-learn
    preds_real = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).ravel() # Achata o array para uma única dimensão 

    return preds_real