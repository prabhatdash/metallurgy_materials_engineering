try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import joblib

    np.random.seed(42)

    data = pd.read_csv('carbon_equv.csv')

    X = data[['C','P','S','Al','Nb','Ti','ceq','a_temp']]
    y = data['r']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
        'Support Vector Regressor': SVR(),
        'kNN': KNeighborsRegressor(n_neighbors=5)
    }


    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = r2 * 100
        results[name] = {'MSE': mse, 'R2': r2, 'Accuracy (%)': accuracy}
        fig_actual_vs_predicted = go.Figure()
        fig_actual_vs_predicted.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted'))
        fig_actual_vs_predicted.update_layout(
            title=f'Actual vs Predicted Values ({name})',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            showlegend=True
        )
        fig_actual_vs_predicted.write_image(f"graphs/{name}_actual_vs_predicted.png")
        fig_actual_vs_predicted.show()

    results_df = pd.DataFrame(results).T
    print(results_df)

    fig_r2 = px.bar(results_df, x=results_df.index, y='R2', title='Model Performance Comparison (R^2 Score)',
                    labels={'index': 'Model', 'R2': 'R^2 Score'})
    fig_r2.write_image("graphs/1_original_data_r2.png")
    fig_r2.show()

    fig_mse = px.bar(results_df, x=results_df.index, y='MSE', title='Model Performance Comparison (MSE)',
                     labels={'index': 'Model', 'MSE': 'Mean Squared Error'})
    fig_mse.write_image("graphs/1_original_data_mse.png")
    fig_mse.show()

    fig_accuracy = px.bar(results_df, x=results_df.index, y='Accuracy (%)',
                          title='Model Performance Comparison (Accuracy Percentage)',
                          labels={'index': 'Model', 'Accuracy (%)': 'Accuracy (%)'})
    fig_accuracy.write_image("graphs/1_original_data_accuracy.png")
    fig_accuracy.show()

    # Generating heatmap for correlation matrix
    corr_matrix = data.corr()
    fig_heatmap = ff.create_annotated_heatmap(z=corr_matrix.values,
                                              x=list(corr_matrix.columns),
                                              y=list(corr_matrix.index),
                                              colorscale='Viridis')
    fig_heatmap.update_layout(title='Correlation Heatmap')
    fig_heatmap.write_image("graphs/heatmap.png")
    fig_heatmap.show()

    best_model_name = results_df['R2'].idxmax()
    best_model = models[best_model_name]
    print(
        f"The best model is: {best_model_name} with R^2 Score: {results_df['R2'].max()} and Accuracy: {results_df['Accuracy (%)'].max()}%")

    joblib.dump(best_model, 'best_model.pkl')

    best_model = joblib.load('best_model.pkl')

    # Example: Using the model to make a prediction
    input_data = np.array([[0.03,0.01,0.004,0.032,0,0,0.318066667,830]])  # Example input
    predicted_ys = best_model.predict(input_data)
    print(f"Predicted Val: {predicted_ys[0]}")

    train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig_learning_curve = go.Figure()
    fig_learning_curve.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training score',
        line=dict(color='red'), fill='tonexty',
        error_y=dict(type='data', array=train_scores_std, visible=True)))
    fig_learning_curve.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Cross-validation score',
        line=dict(color='green'), fill='tonexty',
        error_y=dict(type='data', array=test_scores_std, visible=True)))

    fig_learning_curve.update_layout(
        title=f'Learning Curve for {best_model_name}',
        xaxis_title='Training examples',
        yaxis_title='Score',
        legend_title='Legend'
    )
    fig_learning_curve.write_image("graphs/learning_curve.png")
    fig_learning_curve.show()

except Exception as e:
    print(e)
