from gru_scaler import StandardScaler, MinMaxScaler
from gru_trainer import get_train_test_holdout_data, train_and_test_gru_model
from stock_predictor_models import StockGRU
import pandas as pd

def hyperparameter_search():
    results = []

    # Define search space
    hidden_dims = [32, 64]
    num_layers_list = [1, 2]
    dropouts = [0.0, 0.1, 0.2]
    learning_rates = [1e-3, 5e-4]
    seq_lens = [30, 60]

    scaler_types = [StandardScaler(), MinMaxScaler()]

    for scaler in scaler_types:
        for seq_len in seq_lens:
            # Load + scale data once per sequence length
            X_train, X_test, X_holdout, y_train, y_test, y_holdout = \
                get_train_test_holdout_data(scaler=scaler, seq_len=seq_len)

            for hidden_dim in hidden_dims:
                for layers in num_layers_list:
                    for dropout in dropouts:
                        for lr in learning_rates:

                            print("\n========== RUNNING CONFIG ==========")
                            print(f"seq_len={seq_len}, hidden_dim={hidden_dim}, layers={layers}, "
                                  f"dropout={dropout}, lr={lr}")
                            print("====================================\n")

                            model = StockGRU(
                                input_dim=X_train.shape[-1],
                                hidden_dim=hidden_dim,
                                output_dim=X_train.shape[-1],
                                dropout=dropout,
                                num_layers=layers
                            )

                            _, _, _, metrics = train_and_test_gru_model(
                                model, X_train, y_train, X_test, y_test,
                                lr=lr, epochs=50
                            )

                            # Collect results
                            results.append({
                                "seq_len": seq_len,
                                "hidden_dim": hidden_dim,
                                "layers": layers,
                                "dropout": dropout,
                                "lr": lr,
                                "avg_mse": metrics["average_mse"],
                                "avg_mae": metrics["average mae"],
                                "avg_smape": metrics["average_smape"],
                                "avg_dir_acc": metrics["average directional accuracy"]
                            })

    # convert to DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv("gru_hparam_results.csv", index=False)
    print("\n===== Hyperparameter Search Complete =====\n")
    print(df_results)
    return df_results

if __name__ == '__main__':
    hyperparameter_search()