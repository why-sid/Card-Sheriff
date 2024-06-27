# Card Sheriff

This project allows users to train various machine learning models using different options such as model selection, data balancing, and test size adjustment. It provides visual feedback through particle animations and displays model evaluation metrics including accuracy, F1 score, precision, recall, and a confusion matrix.

## Features

- **Model Selection:** Choose from a variety of classifiers including XGBClassifier, DecisionTreeClassifier, CatBoostClassifier, GradientBoostingClassifier, and RandomForestClassifier.
  
- **Data Balancing:** Option to balance data using SMOTE.
  
- **Test Size Adjustment:** Slider to adjust the test size for model evaluation.

- **Visual Feedback:** Particle animation background that interacts with user input actions.

- **Metrics Display:** Real-time display of model performance metrics including accuracy, F1 score, precision, and recall.

## Usage

1. **Model Selection:** Choose a model from the dropdown menu.
   
2. **Data Balancing:** Check the box to apply SMOTE for data balancing.
   
3. **Test Size:** Adjust the slider to set the test size.

4. **Training:** Click on "Train Model" to initiate the training process. A loading animation appears during training.

5. **Results:** Once training is complete, results including metrics and confusion matrix are displayed.

## Installation

1. Clone the repository:
git clone <https://github.com/why-sid/Card-Sheriff.git>

2. Install the dependencies
pip install -r requirements.txt

3. Run the application:
python app.py

4. Open your web browser and go to `http://localhost:5000`.

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python Flask
- **Visualization:** particles.js for particle animations
- **Machine Learning:** scikit-learn, XGBoost, CatBoost

## Credits

- Developed by [why-sid](https://github.com/why-sid)
- Particle animations powered by particles.js
