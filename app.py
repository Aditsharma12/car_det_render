from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# India-specific pricing factors
BRAND_FACTORS = {
    "maruti": 1.1, "hyundai": 1.05, "tata": 0.95, 
    "mahindra": 0.97, "kia": 0.93, "honda": 0.98,
    "toyota": 1.08, "default": 1.0
}

def calculate_indian_price(base_price, damage, mileage, age, brand):
    brand_factor = BRAND_FACTORS.get(brand.lower(), BRAND_FACTORS["default"])
    age_penalty = min(age * 0.10, 0.7)
    mileage_penalty = min(mileage / 100000, 0.6)
    damage_penalty = damage * 0.8
    price = (base_price * brand_factor * (1 - max(age_penalty, mileage_penalty, damage_penalty))) / 100000
    return max(0.5, round(price, 2))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            base_price = float(request.form['base_price'])
            mileage = int(request.form['mileage'])
            age = int(request.form['age'])
            brand = request.form['brand']
            
            # Process uploaded image
            file = request.files['car_image']
            img = Image.open(io.BytesIO(file.read()))
            
            # Get damage score
            img = img.convert("RGB").resize((640, 640))
            img_array = np.array(img).transpose(2, 0, 1) / 255.0
            session = ort.InferenceSession("damaged_car_classifier.onnx")
            outputs = session.run(None, {"images": np.expand_dims(img_array, 0).astype(np.float32)})[0]
            damage = float(np.max(outputs[0, :, 4])) if outputs.shape[1] > 0 else 0.0
            
            # Calculate price
            price = calculate_indian_price(base_price, damage, mileage, age, brand)
            
            return jsonify({
                'success': True,
                'price': price,
                'damage': f"{damage:.0%}",
                'brand': brand.title()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)