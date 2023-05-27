import pickle
from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from yellowbrick import ROCAUC
from yellowbrick.classifier import ClassPredictionError, ClassificationReport, ConfusionMatrix
from yellowbrick.contrib.wrapper import wrap

CURRENT_DIR = path.dirname(path.realpath(__file__))
INPUT_FILE = path.join(CURRENT_DIR, "input", "demo.csv")
TYPES = ['cat', 'xgb', 'gra']
INCLUDE_ONSET_DATES = [True, False]
LABELS = ['Incidence_leaves', 'Incidence_bunches', 'Severity_bunches', 'Severity_leaves']
RAW_COLUMNS = ['Rainfall Mar', 'Rainfall Apr', 'Rainfall May', 'Rainfall Jun', 'Temp Mar', 'Temp Apr', 'Temp May',
               'Temp Jun']
LABEL_VALUES = [0, 1]
TEMPERATURE_CHANGES = [0, 1, 2, 3, 4]  # in degrees Celsius
RAINFALL_CHANGES = [-15, -10, -5, 0, 5, 10, 15]  # in percentage

# Load the data from CSV file
raw_data = pd.read_csv(INPUT_FILE)
makedirs(path.join(CURRENT_DIR, 'output'), exist_ok=True)


def generate_output_folders(include_offset_date, label, type):
    model_output_path = path.join(CURRENT_DIR, 'output',
                                  type if not include_offset_date else type + '_date', label)
    makedirs(model_output_path, exist_ok=True)
    return model_output_path


def prepare_data(include_offset_date, label, output_path):
    data = raw_data.copy(deep=True)
    columns = RAW_COLUMNS[:]
    if include_offset_date:
        columns.append('Onset Date')
    # Separate features (X) and target variable (y)
    X = data[columns]
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test.to_csv(path.join(output_path, 'X_test.csv'), index=False)
    y_test.to_csv(path.join(output_path, 'y_test.csv'), index=False, header=True)
    return X_train, X_test, y_train, y_test, data, columns, X.columns


def create_and_train_model(X_test, X_train, model_type, y_test, y_train):
    if model_type == 'cat':
        model = CatBoostClassifier(task_type="GPU", devices='0:1')
    elif model_type == 'xgb':
        model = XGBClassifier()
    elif model_type == 'gra':
        model = GradientBoostingClassifier(learning_rate=0.1)
    else:
        raise RuntimeError(f"Model type is needed.")
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions on the testing set
    y_pred = model.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    return model


def write_feature_importances(model, model_output_path, model_type, x_columns):
    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({'Feature': x_columns, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    # Save the plot to a PNG file
    plt.savefig(path.join(model_output_path, 'feature_importances.png'))
    print("Saving model to disk...")
    pickle.dump(model, open(path.join(model_output_path, 'model.sav'), 'wb'))


def wrap_model(model, model_type):
    # Needed for writing reports
    if model_type == 'cat':
        model = wrap(model)
    return model


def write_classification_report(model, X_train, X_test, y_train, y_test, output_path):
    print("Writing classification report...")
    color = ['yellow', 'green', 'brown', 'black']
    vr = ClassificationReport(model, classes=LABEL_VALUES, support=True)
    vr.fit(X_train, y_train)
    vr.score(X_test, y_test)
    vr.show(path.join(output_path, 'Classification_report.png'), clear_figure=True)


def write_classification_prediction_error(model, X_train, X_test, y_train, y_test, output_path):
    print("Writing classification prediction error...")
    vlr = ClassPredictionError(model, classes=LABEL_VALUES)
    vlr.fit(X_train, y_train)
    vlr.score(X_test, y_test)
    vlr.show(path.join(output_path, 'Class_prediction_error.png'), clear_figure=True)


def write_roc_auc_report(model, X_train, X_test, y_train, y_test, output_path):
    print("Writing ROCAUC...")
    visualizer = ROCAUC(model, classes=LABEL_VALUES)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(path.join(output_path, 'ROC_auc.png'), clear_figure=True)


def write_confusion_matrix(model, X_train, X_test, y_train, y_test, output_path):
    print("Writing confusion matrix...")
    cm = ConfusionMatrix(model, classes=LABEL_VALUES)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.show(path.join(output_path, 'Confusion_matrix.png'), clear_figure=True)


def write_auc_score(model, X_test, y_test, output_path):
    print("Writing AUC scores...")
    global catboost_auc, filename
    predictions = model.predict_proba(X_test)[:, 1]
    catboost_auc = roc_auc_score(y_test, predictions)
    filename = path.join(output_path, 'auc_score.txt')
    with open(filename, 'w') as file:
        file.write(str(catboost_auc))


def generate_probabilities(model, data, columns, output_path):
    print("Generating probability data...")
    data = data[columns]
    probabilities = []
    # Iterate through temperature and rainfall changes
    for temperature_change in TEMPERATURE_CHANGES:
        for rainfall_change in RAINFALL_CHANGES:
            modified_dataset = data.copy()
            for column in ['Temp Mar', 'Temp Apr', 'Temp May', 'Temp Jun', 'Rainfall Mar', 'Rainfall Apr',
                           'Rainfall May', 'Rainfall Jun']:
                modified_dataset[column] = modified_dataset[column] + temperature_change if column.startswith(
                    'Temp') else modified_dataset[column] * (1 + rainfall_change / 100)
            # Apply the best model to compute the probability of high levels of GDM
            probability = model.predict_proba(modified_dataset)[:, 1]
            probabilities.append(probability)
    # Create a list to store the reshaped probabilities for each temperature change
    reshaped_probabilities = []
    for i in range(len(TEMPERATURE_CHANGES)):
        start_index = i * len(RAINFALL_CHANGES)
        end_index = start_index + len(RAINFALL_CHANGES)
        reshaped_probabilities.append(probabilities[start_index:end_index])

    # Convert the reshaped probabilities into a numpy array
    probabilities = np.array(reshaped_probabilities)
    generate_variance_plot(probabilities, output_path)


def generate_variance_plot(all_probabilities, output_path):
    print("Generating variance plot...")
    # Create four separate plots for each temperature change
    fig, axes = plt.subplots(1, len(TEMPERATURE_CHANGES), figsize=(16, 4))
    # Iterate over each temperature change
    for i, temperature in enumerate(TEMPERATURE_CHANGES):
        ax = axes[i]
        list = all_probabilities[i]
        d = []
        for l in list:
            d.append(l)
        # x-axis labels
        ax.set_xticklabels(RAINFALL_CHANGES)
        ax.title.set_text('Temperature Change: {}°C'.format(temperature))
        ax.set_ylim([-0.05, 1.05])
        bp = ax.boxplot(d)
        for median in bp['medians']:
            median.set(color='red', linewidth=3)
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(path.join(output_path, 'variances.png'))


def main():
    print("==================================================")
    print("WELCOME TO MODEL SIMULATOR (AUTHOR: DR. KAI-YUN LI")
    print("==================================================")

    for m_type in TYPES:
        for label in LABELS:
            for include_offset_date in INCLUDE_ONSET_DATES:
                model_output_path = generate_output_folders(include_offset_date, label, m_type)
                X_train, X_test, y_train, y_test, data, columns, x_columns = prepare_data(include_offset_date, label,
                                                                                          model_output_path)
                model = create_and_train_model(X_test, X_train, m_type, y_test, y_train)
                write_feature_importances(model, model_output_path, m_type, x_columns)
                model = wrap_model(model, m_type)
                write_classification_report(model, X_train, X_test, y_train, y_test, model_output_path)
                write_classification_prediction_error(model, X_train, X_test, y_train, y_test, model_output_path)
                write_roc_auc_report(model, X_train, X_test, y_train, y_test, model_output_path)
                write_confusion_matrix(model, X_train, X_test, y_train, y_test, model_output_path)
                write_auc_score(model, X_test, y_test, model_output_path)
                generate_probabilities(model, data, columns, model_output_path)


if __name__ == '__main__':
    main()
