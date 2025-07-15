import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load macroeconomic data
macro_df = pd.read_csv('MacroEconomicDataFrame.csv')

# Load feature explanations from the uploaded CSV file (used for interpreting LIME features)
@st.cache_data
def load_feature_explanations():
    df = pd.read_csv("Feature_Comments_and_Suggestions.csv")
    return {
        row['feature']: {
            'label': row['label'],
            'positive': row['positive'],
            'negative': row['negative']
        }
        for _, row in df.iterrows()
    }

# Dictionary for interpreting LIME top features
comments_dict = load_feature_explanations()

# Encoding map
encoding_maps = {
    'AgeAtEnrollment': {'18 - 20 years': 1, '21 - 23 years': 2, '24 - 26 years': 3, '26 years and above': 4},
    'SyllabusMedium': ['Local Government Syllabus (Sri Lankan : English)',
                       'Local Government Syllabus (Sri Lankan : Tamil)',
                       'Local Government Syllabus (Sri Lankan : Sinhala)',
                       'Other', 'Cambridge International (CIE) or Edexcel (Pearson)'],
    'OLevelCoreModulePass': {
        "Yes, I achieved a minimum 'C' pass in all three subjects": 1,
        "No, I did not achieve a minimum 'C' pass in one or more of these subjects": 0
    },
    'ALStream': ['Mathematics Stream', 'Commerce Stream', 'Science Stream',
                 "Didn't take the A - Level examinations", 'Technology Stream', 'Arts Stream'],
    'ALevelCoreModulePass': {'Yes': 1, 'No': 0},
    'ALEnglishOrCourse': ['No, I did not follow A-Level English or any English course',
                          'Yes, I studied A-Level English',
                          'Yes, I completed an English course',
                          'Yes, I followed both A-Level English and completed an English course'],
    'PriorHigherEdu': {'No, I did not pursue any higher education prior to this degree': 0,
                       'Foundation Program / Diploma related to Information Technology': 1},
    'GraduationYear': {
        2025: 1, 2026: 2, 2027: 3, 2028: 4, 2029: 5,
        2030: 6, 2031: 7, 2032: 8, 2033: 9, 2034: 10, 2035: 11
    },
    'SecondYearAvg': {'Below 40%': 1, '40% - 50%': 2, '51% - 60%': 3, '61% - 70%': 4, 'Above 70%': 5},
    'InternshipCompleted': {'Yes, I completed the recommended internship': 1,
                            'No, I did not complete the recommended internship': 0},
    'SatisfactionRating': [1, 2, 3, 4, 5],
    'FinalClassification': {'Pass Class': 1, 'Second Class Lower': 2, 'Second Class Upper': 3, 'First Class': 4},
    'StressAnxietyLevel': [1, 2, 3, 4, 5],
    'PhysicalHealth': [1, 2, 3, 4, 5],
    'ChronicIllness': {'No': 0, 'Yes': 1},
    'ParentsEmployment': ['Both parents/guardians are employed',
                          'One parent/guardian is employed',
                          'Neither parents/guardian is employed'],
    'ParentsEducation': ["Higher education (Diploma or Bachelor's degree)",
                         "Postgraduate education (Master's degree or higher)",
                         'Completed A-Level', 'Completed O-Level', 'Prefer not to say'],
    'ParentsCohabitation': ['Both parents/guardians live together',
                            'Parents/guardians are separated but living independently',
                            'Both parents/guardians are deceased'],
    'HouseholdIncome': {'Below LKR 100,000': 1, 'LKR 100,000 - 300,000': 2, 'LKR 300,000 - 500,000': 3,
                        'Above LKR 500,000': 4, 'Prefer not to say': 0},
    'AccommodationType': ['Living with parents/guardians',
                          'Off-campus rented accommodation',
                          'Shared accommodation with friends or relatives'],
    'TransportMode': ['Driving / Driven by personal vehicle',
                      'Uber or other ride-hailing services',
                      'Public bus / train'],
    'TravelTime': {'Less than 30 minutes': 1, '30 minutes to 1 hour': 2, '1 hour to 1.5 hours': 3,
                   '1.5 hours to 2 hours': 4, 'More than 2 hours': 5},
    'EmployedDuringDegree': ['Yes, full-time employment', 'Yes, part-time employment', 'No, I was not employed'],
    'LeisureHoursPerWeek': {'Less than 5 hours': 1, '05 – 10 hours': 2, '10 – 20 hours': 3, 'More than 20 hours': 4},
    'DailyScreenTime': {'Less than 2 hours': 1, '2 – 4 hours': 2, '5 – 7 hours': 3, '8 – 10 hours': 4, 'More than 10 hours': 5}
}

def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]

def multi_label_encode(selected, categories):
    return [1 if cat in selected else 0 for cat in categories]

def get_macro_features(grad_year):
    row = macro_df[macro_df['Graduation Year'] == grad_year]
    if not row.empty:
        return (
            row.iloc[0]['Inflation Rate Percentage Point Change'],
            row.iloc[0]['Unemployment Rate Percentage Point Change'],
            row.iloc[0]['Exchange Rate Percentage Point Change']
        )
    return 0.0, 0.0, 0.0

def encode_input(user_input, macro_values):
    encoded = []
    encoded.append(encoding_maps['AgeAtEnrollment'][user_input['AgeAtEnrollment']])
    encoded.extend(one_hot_encode(user_input['SyllabusMedium'], encoding_maps['SyllabusMedium']))
    encoded.append(encoding_maps['OLevelCoreModulePass'][user_input['OLevelCoreModulePass']])
    encoded.extend(one_hot_encode(user_input['ALStream'], encoding_maps['ALStream']))
    encoded.append(encoding_maps['ALevelCoreModulePass'][user_input['ALevelCoreModulePass']])
    encoded.extend(one_hot_encode(user_input['ALEnglishOrCourse'], encoding_maps['ALEnglishOrCourse']))
    encoded.append(encoding_maps['PriorHigherEdu'][user_input['PriorHigherEdu']])
    graduation_year = int(user_input['GraduationYear'])
    encoded.append(encoding_maps['GraduationYear'].get(graduation_year, 0))
    encoded.append(encoding_maps['SecondYearAvg'][user_input['SecondYearAvg']])
    encoded.append(encoding_maps['InternshipCompleted'][user_input['InternshipCompleted']])
    encoded.append(int(user_input['SatisfactionRating']))
    encoded.append(int(user_input['StressAnxietyLevel']))
    encoded.append(int(user_input['PhysicalHealth']))
    encoded.append(encoding_maps['ChronicIllness'][user_input['ChronicIllness']])
    encoded.extend(one_hot_encode(user_input['ParentsEmployment'], encoding_maps['ParentsEmployment']))
    encoded.extend(one_hot_encode(user_input['ParentsEducation'], encoding_maps['ParentsEducation']))
    encoded.extend(one_hot_encode(user_input['ParentsCohabitation'], encoding_maps['ParentsCohabitation']))
    encoded.append(encoding_maps['HouseholdIncome'][user_input['HouseholdIncome']])
    encoded.extend(one_hot_encode(user_input['AccommodationType'], encoding_maps['AccommodationType']))
    encoded.extend(multi_label_encode(user_input['TransportMode'], encoding_maps['TransportMode']))
    encoded.append(encoding_maps['TravelTime'][user_input['TravelTime']])
    encoded.extend(one_hot_encode(user_input['EmployedDuringDegree'], encoding_maps['EmployedDuringDegree']))
    encoded.append(encoding_maps['LeisureHoursPerWeek'][user_input['LeisureHoursPerWeek']])
    encoded.append(encoding_maps['DailyScreenTime'][user_input['DailyScreenTime']])
    encoded.extend(list(macro_values))
    return np.array(encoded).reshape(1, -1)

def validate_inputs(inputs, transport_modes, valid_modes):
    if any(value == 'Select an option' for key, value in inputs.items() if isinstance(value, str)):
        return False
    if not transport_modes or any(mode not in valid_modes for mode in transport_modes):
        return False
    return True

def main():
    st.title('Academic Degree Classification Predictor')

    def select(label, options):
        if isinstance(options, dict):
            options = list(options.keys())
        return st.selectbox(label, ['Select an option'] + options)

    def select_keyed(label, mapping):
        return st.selectbox(label, ['Select an option'] + list(mapping.keys()))

    AgeAtEnrollment = select_keyed('01. What was your age when you enrolled in your current degree program?', encoding_maps['AgeAtEnrollment'])
    SyllabusMedium = select('02. Which syllabus and medium did you follow during your primary and secondary education?', encoding_maps['SyllabusMedium'])
    OLevelCoreModulePass = select_keyed('03. Did you pass Mathematics, English, and Computing at O-Level with at least a grade C?', encoding_maps['OLevelCoreModulePass'])
    ALStream = select('04. Which stream did you study for your A-Level examination?', encoding_maps['ALStream'])
    ALevelCoreModulePass = select_keyed('05. Did you achieve at least three passes with a grade C or higher in your A-Level exams?', encoding_maps['ALevelCoreModulePass'])
    ALEnglishOrCourse = select('06. Did you take A-Level English or complete an English course before starting your degree program?', encoding_maps['ALEnglishOrCourse'])
    PriorHigherEdu = select_keyed('07. Did you pursue any higher education program before starting your current degree?', encoding_maps['PriorHigherEdu'])
    GraduationYear_input = select_keyed('08. In which year will you graduate from the Informatics Institute of Technology (IIT) Sri Lanka?', encoding_maps['GraduationYear'])
    SecondYearAvg = select_keyed('09. What was your average percentage during your second year?', encoding_maps['SecondYearAvg'])
    InternshipCompleted = select_keyed('10. Did you complete the recommended internship during your third year?', encoding_maps['InternshipCompleted'])
    SatisfactionRating = select('11. How satisfied were you with your degree program and career path? (Scale: 1 to 5, with 1 being the lowest and 5 the highest)', list(map(str, encoding_maps['SatisfactionRating'])))
    StressAnxietyLevel = select('12. How would you rate your stress and anxiety levels during your degree? (Scale: 1 to 5, with 1 being the lowest and 5 the highest)', list(map(str, encoding_maps['StressAnxietyLevel'])))
    PhysicalHealth = select('13. How would you rate your physical health during your degree? (Scale: 1 to 5, with 1 being the lowest and 5 the highest)', list(map(str, encoding_maps['PhysicalHealth'])))
    ChronicIllness = select_keyed('14. Did you have any chronic illnesses during your degree?', encoding_maps['ChronicIllness'])
    ParentsEmployment = select('15. What is the employment status of your parents or guardians?', encoding_maps['ParentsEmployment'])
    ParentsEducation = select('16. What is the highest education level of your parents/guardians?', encoding_maps['ParentsEducation'])
    ParentsCohabitation = select('17. What is the current cohabitation status of your parents?', encoding_maps['ParentsCohabitation'])
    HouseholdIncome = select_keyed('18. What is your average monthly household income?', encoding_maps['HouseholdIncome'])
    AccommodationType = select('19. What type of accommodation do you currently live in?', encoding_maps['AccommodationType'])
    TransportMode = st.multiselect('20. What is your main mode of transportation to attend lectures?', encoding_maps['TransportMode'])
    TravelTime = select_keyed('21. On average, how much time do you spend traveling?', encoding_maps['TravelTime'])
    EmployedDuringDegree = select('22. Were you employed during your degree program?', encoding_maps['EmployedDuringDegree'])
    LeisureHoursPerWeek = select_keyed('23. How many hours per week do you spend on leisure activities?', encoding_maps['LeisureHoursPerWeek'])
    DailyScreenTime = select_keyed('24. How many hours per day do you spend on non-study screen time?', encoding_maps['DailyScreenTime'])

    if st.button('Predict Degree Classification'):
        if GraduationYear_input == 'Select an option':
            st.error('Please select a valid graduation year.')
            return

        GraduationYear = int(GraduationYear_input)

        inputs = {
            'AgeAtEnrollment': AgeAtEnrollment,
            'SyllabusMedium': SyllabusMedium,
            'OLevelCoreModulePass': OLevelCoreModulePass,
            'ALStream': ALStream,
            'ALevelCoreModulePass': ALevelCoreModulePass,
            'ALEnglishOrCourse': ALEnglishOrCourse,
            'PriorHigherEdu': PriorHigherEdu,
            'GraduationYear': GraduationYear,
            'SecondYearAvg': SecondYearAvg,
            'InternshipCompleted': InternshipCompleted,
            'SatisfactionRating': SatisfactionRating,
            'StressAnxietyLevel': StressAnxietyLevel,
            'PhysicalHealth': PhysicalHealth,
            'ChronicIllness': ChronicIllness,
            'ParentsEmployment': ParentsEmployment,
            'ParentsEducation': ParentsEducation,
            'ParentsCohabitation': ParentsCohabitation,
            'HouseholdIncome': HouseholdIncome,
            'AccommodationType': AccommodationType,
            'TransportMode': TransportMode,
            'TravelTime': TravelTime,
            'EmployedDuringDegree': EmployedDuringDegree,
            'LeisureHoursPerWeek': LeisureHoursPerWeek,
            'DailyScreenTime': DailyScreenTime
        }

        if not validate_inputs(inputs, TransportMode, encoding_maps['TransportMode']):
            st.error('Please fill all fields before prediction.')
            return

        try:
            macro_values = get_macro_features(inputs['GraduationYear'])
            encoded_input = encode_input(inputs, macro_values)
            scaled_input = scaler.transform(encoded_input)
            prediction = model.predict(scaled_input)

            inverse_map = {v: k for k, v in encoding_maps['FinalClassification'].items()}
            predicted_label = inverse_map.get(prediction[0], 'Unknown')

            st.success(f'Predicted degree classification: {predicted_label}')

            st.write('### Macroeconomic Factor Percentage Point Change (Year of Enrollment to Year of Graduation)')
            st.write(f"**Inflation Rate Change:** {macro_values[0]} percentage points")
            st.write(f"**Unemployment Rate Change:** {macro_values[1]} percentage points")
            st.write(f"**Exchange Rate Change:** {macro_values[2]} percentage points")

            feature_names = ['AgeAtEnrollment'] + \
                            [f'SyllabusMedium_{s}' for s in encoding_maps['SyllabusMedium']] + \
                            ['OLevelCoreModulePass'] + \
                            [f'ALStream_{s}' for s in encoding_maps['ALStream']] + \
                            ['ALevelCoreModulePass'] + \
                            [f'ALEnglishOrCourse_{s}' for s in encoding_maps['ALEnglishOrCourse']] + \
                            ['PriorHigherEdu', 'GraduationYear', 'SecondYearAvg', 'InternshipCompleted',
                             'SatisfactionRating', 'StressAnxietyLevel', 'PhysicalHealth', 'ChronicIllness'] + \
                            [f'ParentsEmployment_{s}' for s in encoding_maps['ParentsEmployment']] + \
                            [f'ParentsEducation_{s}' for s in encoding_maps['ParentsEducation']] + \
                            [f'ParentsCohabitation_{s}' for s in encoding_maps['ParentsCohabitation']] + \
                            ['HouseholdIncome'] + \
                            [f'AccommodationType_{s}' for s in encoding_maps['AccommodationType']] + \
                            [f'TransportMode_{s}' for s in encoding_maps['TransportMode']] + \
                            ['TravelTime'] + \
                            [f'EmployedDuringDegree_{s}' for s in encoding_maps['EmployedDuringDegree']] + \
                            ['LeisureHoursPerWeek', 'DailyScreenTime',
                             'InflationRateChange', 'UnemploymentRateChange', 'ExchangeRateChange']

            dummy_data = np.random.normal(loc=0.0, scale=1.0, size=(100, scaled_input.shape[1]))

            explainer = LimeTabularExplainer(
                training_data=dummy_data,
                feature_names=feature_names,
                class_names=list(encoding_maps['FinalClassification'].keys()),
                mode='classification'
            )

            explanation = explainer.explain_instance(
                scaled_input[0],
                model.predict_proba,
                num_features=3
            )

            st.markdown('<h3 style="font-size:24px;">Prediction Probabilities</h3>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 10))
            probs = model.predict_proba(scaled_input)[0]
            classes = list(encoding_maps['FinalClassification'].keys())

            ax.barh(classes, probs, color='red')
            ax.set_xlabel("Probability", fontsize=12)
            ax.set_ylabel("Classification", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_yticklabels(classes, fontsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.invert_yaxis()
            ax.spines[['top', 'right']].set_visible(False)

            st.pyplot(fig)

            st.markdown('<h3 style="font-size:24px;">Explanation Summary</h3>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:16px;">The model predicted <strong>{predicted_label}</strong> based on the most influential features listed below. '
                f'These factors had the greatest local impact on determining this classification result.</p>',
                unsafe_allow_html=True
            )

            st.markdown('<h3 style="font-size:24px;">Top 03 Feature Contributions</h3>', unsafe_allow_html=True)

            count = 0
            for raw_feature, weight in explanation.as_list():
                if count >= 3:
                    break

                clean_feature = raw_feature.split('<=')[0].split('>')[0].strip()
                info = comments_dict.get(clean_feature)

                if info is None:
                    for key in comments_dict:
                        if key in clean_feature:
                            info = comments_dict[key]
                            break

                if info:
                    label = info['label']
                    direction = 'positive' if weight > 0 else 'negative'
                    suggestion = info[direction]
                    sign = "↑ helped" if weight > 0 else "↓ reduced"

                    st.markdown(f"**{label}**")
                    st.markdown(f"→ {sign} your predicted classification *(Impact: {weight:+.4f})*")
                    st.markdown(f"**Suggestion:** {suggestion}")
                    st.markdown("---")
                    count += 1

            st.markdown('<h3 style="font-size:24px;">Disclaimer</h3>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:16px;">Predictions are based on a limited dataset and may not generalize beyond the original study scope. '
                'Results should be used for guidance only and not as definitive academic evaluations.</p>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()
