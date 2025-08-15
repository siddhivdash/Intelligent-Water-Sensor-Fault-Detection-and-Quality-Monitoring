from flask import Flask, request, render_template
import os
import joblib

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logger

application = Flask(__name__)
app = application


def load_sensor_labels():
    """Loads labels from calibration_params.pkl, defaults if not available."""
    try:
        calib_path = os.path.join('artifacts', 'calibration_params.pkl')
        calibration_params = joblib.load(calib_path)
        property_names = {
            "Sensor-1": "pH",
            "Sensor-2": "Turbidity",
            "Sensor-3": "Conductivity",
            "Sensor-4": "Dissolved Oxygen",
            "Sensor-5": "Chlorine Level",
            "Sensor-6": "Nitrate",
            "Sensor-7": "Hardness",
            "Sensor-8": "Temperature",
            "Sensor-9": "Iron Content",
            "Sensor-10": "BOD"
        }
        sensor_labels = []
        for i in range(1, 11):
            key = f"Sensor-{i}"
            if key in calibration_params:
                ymin = calibration_params[key].get("ymin", 0)
                ymax = calibration_params[key].get("ymax", 0)
                prop_name = property_names.get(key, key)
                label = f"{prop_name} ({ymin} - {ymax})"
                sensor_labels.append(label)
            else:
                sensor_labels.append(f"Sensor-{i}")
        logger.info(f"Sensor labels loaded: {sensor_labels}")
        return sensor_labels
    except Exception as e:
        logger.error(f"Error loading calibration params: {e}")
        return [f"Sensor-{i}" for i in range(1, 11)]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    sensor_labels = load_sensor_labels()

    if request.method == 'GET':
        return render_template('home.html', results=None, error_message=None, sensor_labels=sensor_labels)

    else:
        try:
            # 📌 Step 1: Collect inputs
            inputs = [
                float(request.form.get('sensor_1')),  # pH
                float(request.form.get('sensor_2')),  # Turbidity
                float(request.form.get('sensor_3')),  # Conductivity
                float(request.form.get('sensor_4')),  # Dissolved Oxygen
                float(request.form.get('sensor_5')),  # Chlorine Level
                float(request.form.get('sensor_6')),  # Nitrate
                float(request.form.get('sensor_7')),  # Hardness
                float(request.form.get('sensor_8')),  # Temperature
                float(request.form.get('sensor_9')),  # Iron Content
                float(request.form.get('sensor_10'))  # BOD
            ]

            # 📌 Step 2: Define valid ranges (ymin, ymax from calibration)
            limits = [
                (0, 14),    # pH
                (0, 100),   # Turbidity
                (0, 2000),  # Conductivity
                (0, 50),    # Dissolved Oxygen
                (0, 10),    # Chlorine Level
                (0, 50),    # Nitrate
                (0, 14),    # Hardness
                (0, 500),   # Temperature
                (0, 200),   # Iron Content
                (0, 10)     # BOD
            ]

            # 📌 Step 3: Out-of-range check
            for (val, (mn, mx)) in zip(inputs, limits):
                if val < mn or val > mx:
                    logger.warning(f"Out-of-range value detected: {val} not in ({mn}, {mx})")
                    return render_template(
                        'home.html',
                        results="Faulty Water Sensor (out of range values)",
                        error_message=None,
                        sensor_labels=sensor_labels
                    )

            # 📌 Step 4: Prepare for prediction if all OK
            data = CustomData(
                sensor_1=inputs[0],
                sensor_2=inputs[1],
                sensor_3=inputs[2],
                sensor_4=inputs[3],
                sensor_5=inputs[4],
                sensor_6=inputs[5],
                sensor_7=inputs[6],
                sensor_8=inputs[7],
                sensor_9=inputs[8],
                sensor_10=inputs[9]
            )

            pred_df = data.get_data_as_data_frame()
            pred_df.columns = [f"Sensor-{i}" for i in range(1, 11)]
            logger.info(f"Prediction input DataFrame:\n{pred_df}")

            # 📌 Step 5: Run prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            prediction_text = "Good Water Sensor" if results[0] == 1 else "Faulty Water Sensor"

            return render_template('home.html', results=prediction_text, error_message=None, sensor_labels=sensor_labels)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render_template('home.html', results=None, error_message=str(e), sensor_labels=sensor_labels)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
