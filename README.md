# DegreeClassificationPredictorS

A Streamlit-based machine learning app that predicts undergraduate degree classifications using a combination of academic, socioeconomic, and psychological factors, along with macroeconomic indicators like inflation and unemployment trends. 

Built with a trained Support Vector Machine (SVM) model, this tool offers transparent predictions using **LIME** to explain which features had the most influence. The goal is to help students, educators, and institutions better understand performance drivers and provide targeted academic guidance.

## Key Features

- Predicts one of four final degree outcomes: *Pass Class*, *Second Class Lower*, *Second Class Upper*, or *First Class*
- User-friendly questionnaire interface powered by Streamlit
- Integrates macroeconomic data (inflation, exchange rate, unemployment)
- LIME-based local explanations with actionable suggestions per feature
- Visual output of prediction probabilities and top influencing factors

## Usage

1. Clone the repo and install requirements  
2. Run the app with `streamlit run "Streamlit Prediction Application File New.py"`  
3. Fill in the 24-question form and click **Predict**  
4. Review the prediction, visualized probabilities, and top explanatory features

> Requires: `svm_model.pkl`, `scaler.pkl`, `MacroEconomicDataFrame.csv`, and `Feature_Comments_and_Suggestions.csv`

## Disclaimer

This tool is intended for academic research and guidance purposes only. Predictions are based on a specific dataset and may not generalize to all contexts.

