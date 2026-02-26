from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 8: The Experiment"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 8: The Experiment", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"All models are wrong, but some are useful.\" — George Box</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 8.1 Introduction
    story.append(Paragraph("8.1 The Predictive Engine", styles['SectionTitle']))
    story.append(Paragraph("""
    With our features engineered (Chapter 7), we enter the crucible of machine learning. Our objective is to learn a mapping function $f: X \\rightarrow Y$, where $X$ is our feature vector (RSI, MACD, etc.) and $Y$ is the future price return.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    We selected <b>XGBoost (Extreme Gradient Boosting)</b> as our champion algorithm. It is the de facto standard for tabular data in Kaggle competitions due to its handling of non-linear relationships, built-in regularization, and parallel processing capabilities.
    """, styles['BodyTextCustom']))

    # 8.2 XGBoost Architecture
    story.append(Paragraph("8.2 Algorithm Architecture: Gradient Boosting", styles['SectionTitle']))
    story.append(Paragraph("""
    XGBoost is an ensemble method that builds trees sequentially. Each new tree attempts to correct the errors (residuals) of the previous trees.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>The Objective Function:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    $$Obj(\Theta) = L(\Theta) + \Omega(\Theta)$$
    Where $L$ is the loss function (Mean Squared Error) and $\Omega$ is the regularization term to control complexity.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    Key components:
    -   <b>Tree Pruning:</b> Uses 'max_depth' to prevent overfitting.
    -   <b>Regularization:</b> L1 (Lasso) and L2 (Ridge) penalties on leaf weights.
    -   <b>Shrinkage:</b> Learning rate ($\eta$) scales the contribution of each tree.
    """, styles['BodyTextCustom']))

    # 8.3 Training Pipeline
    story.append(Paragraph("8.3 The Training Pipeline", styles['SectionTitle']))
    story.append(Paragraph("""
    Our training protocol is designed to simulate real-world trading conditions and avoid 'look-ahead bias'.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>Time-Series Cross-Validation:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    Unlike standard K-Fold CV, we cannot shuffle time-series data. We use a rolling window approach:
    -   <b>Fold 1:</b> Train [Jan-Mar], Test [Apr]
    -   <b>Fold 2:</b> Train [Jan-Apr], Test [May]
    -   <b>Fold 3:</b> Train [Jan-May], Test [Jun]
    This ensures we always predict the 'future' based on the 'past'.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: Model Training
    import xgboost as xgb
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    """, styles['CodeBlock']))

    # 8.4 Hyperparameter Tuning
    story.append(Paragraph("8.4 Hyperparameter Optimization", styles['SectionTitle']))
    story.append(Paragraph("""
    A model is only as good as its configuration. We employ <b>Bayesian Optimization</b> (via Optuna) to navigate the hyperparameter space efficiently. Unlike Grid Search (brute force) or Random Search, Bayesian Optimization builds a probabilistic model of the objective function to select the most promising hyperparameters to evaluate next.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>Optimized Parameters:</b>
    -   <b>learning_rate:</b> 0.01 (Slow learning for better generalization)
    -   <b>max_depth:</b> 5 (Limits interaction complexity)
    -   <b>subsample:</b> 0.7 (Stochastic gradient boosting to reduce variance)
    """, styles['InsightBox']))

    story.append(PageBreak())
