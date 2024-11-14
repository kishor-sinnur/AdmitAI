
# AdmitAI - Graduate Admission Prediction Model

AdmitAI is a machine learning project that predicts the chances of a student’s admission based on academic and research-related parameters. Using a neural network built with TensorFlow and Keras, this project provides an estimate of the likelihood of admission to a graduate program.

## Project Overview

This model uses data related to GRE scores, TOEFL scores, university rating, statement of purpose (SOP) and letter of recommendation (LOR) strengths, GPA, and research experience to predict the chance of admission. The dataset is preprocessed, split, and scaled before training the neural network to improve prediction accuracy.

## Dataset

The dataset used in this project has the following features:

- **GRE Score**: Graduate Record Examination score (out of 340)
- **TOEFL Score**: Test of English as a Foreign Language score (out of 120)
- **University Rating**: University ranking (out of 5)
- **SOP**: Strength of Statement of Purpose (out of 5)
- **LOR**: Strength of Letter of Recommendation (out of 5)
- **CGPA**: Cumulative GPA (out of 10)
- **Research**: Research experience (0 or 1, where 1 represents having research experience)
- **Chance of Admit**: Probability of admission (0 to 1)

## Project Structure

- `Admission_Predict.csv`: Contains the dataset for training the model.
- `notebook`: Jupyter notebook with code for data preprocessing, model training, and evaluation.

## Model Architecture

The neural network model is built using TensorFlow's Keras API:

1. **Input Layer**: Takes in the features (6 input nodes).
2. **Hidden Layers**: 
   - Dense layer with 64 nodes and ReLU activation.
   - Dense layer with 32 nodes and ReLU activation.
   - Dense layer with 16 nodes and ReLU activation.
3. **Output Layer**: Dense layer with 1 node for regression output.

### Model Compilation

The model is compiled with:
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam (learning rate = 0.001)
- **Metrics**: Mean Absolute Error

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kishor-sinnur/AdmitAI.git
   ```

2. Navigate into the project directory:

   ```bash
   cd AdmitAI
   ```

3. Install required Python libraries:

   ```bash
   pip install numpy pandas scikit-learn tensorflow
   ```

## Usage

1. **Load the Dataset**:
   ```python
   data = pd.read_csv('Admission_Predict.csv')
   ```

2. **Train the Model**:
   - The model is trained on 80% of the data and tested on the remaining 20%.

3. **Predict Admission Chance for New Data**:
   ```python
   new_student = np.array([[330, 115, 4, 4.5, 4.5, 9.0, 1]])
   new_student_scaled = scaler.transform(new_student)
   predicted_chance = model.predict(new_student_scaled)
   print(f"Predicted Chance of Admit: {predicted_chance[0][0]:.2f}")
   ```

## Sample Predictions

| GRE Score | TOEFL Score | University Rating | SOP | LOR | CGPA | Research | Predicted Chance of Admit |
|-----------|-------------|-------------------|-----|-----|------|----------|----------------------------|
| 330       | 115         | 4                 | 4.5 | 4.5 | 9.0  | 1        | 0.88                       |

## License

This project is licensed under the MIT License.

## Contact

For any queries, please reach out to **Kishor Sinnur** at `kishorsinnuR31GMAIL.COM`.
