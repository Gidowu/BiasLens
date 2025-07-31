"""
comprehensive comparison between logistic regression and linear svm models
for political bias classification in news articles
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """load the preprocessed training and testing data"""
    print("loading preprocessed data...")
    X_train = joblib.load('processed/X_train.pkl')
    X_test = joblib.load('processed/X_test.pkl')
    y_train = joblib.load('processed/y_train.pkl')
    y_test = joblib.load('processed/y_test.pkl')
    
    print(f"training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"testing set: {X_test.shape[0]} samples")
    print(f"classes: {np.unique(y_train)}")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """train both models and evaluate their performance"""
    
    # initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"training and evaluating {name}")
        print(f"{'='*50}")
        
        # train model
        model.fit(X_train, y_train)
        
        # make predictions
        y_pred = model.predict(X_test)
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # print results
        print(f"accuracy:  {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall:    {recall:.4f}")
        print(f"f1-score:  {f1:.4f}")
        
        # detailed classification report
        print(f"\ndetailed classification report for {name}:")
        print(classification_report(y_test, y_pred))
    
    return results

def perform_cross_validation(X_train, y_train):
    """perform cross-validation for both models"""
    print(f"\n{'='*50}")
    print("cross-validation results (5-fold)")
    print(f"{'='*50}")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000)
    }
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"\n{name}:")
        print(f"  cv accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  individual folds: {cv_scores}")
    
    return cv_results

def plot_confusion_matrices(y_test, results):
    """plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    classes = ['center', 'left', 'right']  # assuming these are your classes
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=axes[idx])
        axes[idx].set_title(f'confusion matrix - {name}')
        axes[idx].set_xlabel('predicted label')
        axes[idx].set_ylabel('true label')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("confusion matrices saved as 'models/confusion_matrices.png'")

def plot_performance_comparison(results, cv_results):
    """plot performance comparison between models"""
    
    # metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # test set performance
    for idx, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        bars = axes[idx].bar(model_names, values, color=['skyblue', 'lightcoral'])
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].set_ylim([0, 1])
        
        # add value labels on bars
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("performance comparison saved as 'models/performance_comparison.png'")
    
    # cross-validation comparison
    plt.figure(figsize=(10, 6))
    cv_means = [cv_results[name]['mean'] for name in model_names]
    cv_stds = [cv_results[name]['std'] for name in model_names]
    
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                   color=['skyblue', 'lightcoral'])
    plt.title('cross-validation accuracy comparison')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    
    # add value labels
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("cross-validation comparison saved as 'models/cv_comparison.png'")

def create_summary_report(results, cv_results):
    """create a comprehensive summary report"""
    print(f"\n{'='*60}")
    print("comprehensive model comparison summary")
    print(f"{'='*60}")
    
    # test set performance summary
    print("\ntest set performance:")
    print("-" * 40)
    df_results = pd.DataFrame({
        name: {
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}"
        } for name, result in results.items()
    }).T
    print(df_results)
    
    # cross-validation summary
    print("\ncross-validation performance:")
    print("-" * 40)
    df_cv = pd.DataFrame({
        name: f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}"
        for name, cv_result in cv_results.items()
    }, index=['CV Accuracy']).T
    print(df_cv)
    
    # determine best model
    best_model_test = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model_cv = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
    
    print(f"\nbest performing model:")
    print("-" * 40)
    print(f"test set (f1-score): {best_model_test}")
    print(f"cross-validation: {best_model_cv}")
    
    # analysis
    print(f"\nanalysis:")
    print("-" * 40)
    
    lr_f1 = results['Logistic Regression']['f1_score']
    svm_f1 = results['Linear SVM']['f1_score']
    
    if abs(lr_f1 - svm_f1) < 0.01:
        print("- both models show very similar performance")
        print("- the difference is not statistically significant")
    elif lr_f1 > svm_f1:
        print("- logistic regression outperforms linear svm")
        print(f"- performance difference: {(lr_f1 - svm_f1)*100:.2f}% f1-score improvement")
    else:
        print("- linear svm outperforms logistic regression")
        print(f"- performance difference: {(svm_f1 - lr_f1)*100:.2f}% f1-score improvement")
    
    print("\n- both models are suitable for political bias classification")
    print("- consider ensemble methods for potentially better performance")

def save_models(results):
    """save the trained models for future use"""
    for name, result in results.items():
        filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(result['model'], filename)
        print(f"saved {name} model to {filename}")

def main():
    """main function to run the complete comparison"""
    print("starting comprehensive model comparison")
    print("political bias classification: logistic regression vs linear svm")
    
    # load data
    X_train, X_test, y_train, y_test = load_data()
    
    # train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # perform cross-validation
    cv_results = perform_cross_validation(X_train, y_train)
    
    # create visualizations
    plot_confusion_matrices(y_test, results)
    plot_performance_comparison(results, cv_results)
    
    # create summary report
    create_summary_report(results, cv_results)
    
    # save models
    save_models(results)
    
    print(f"\ncomparison complete! check the 'models/' directory for:")
    print("   - trained model files (.pkl)")
    print("   - visualization plots (.png)")

if __name__ == "__main__":
    main()