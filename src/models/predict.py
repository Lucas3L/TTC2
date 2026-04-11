import numpy as np


def predict_model(model, X_seq, scaler_y, round_to_int=False):
    # Executa a inferência do modelo sobre as sequências
    preds_scaled = model.predict(X_seq, verbose=0)

    # Inverte a normalização para retornar aos valores originais de venda
    preds_real = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).ravel()

    # Garante que não existam valores negativos residuais
    preds_real = np.maximum(0, preds_real)

    # Opcional: arredonda para inteiro
    if round_to_int:
        preds_real = np.round(preds_real).astype(int)

    return preds_real