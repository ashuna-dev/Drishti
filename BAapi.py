from flask import Flask, request, jsonify
from BlindAssistant import BlindAssistant

app = Flask(__name__)
blind_assistant = BlindAssistant()

# Route to start the BlindAssistant
@app.route('/start', methods=['POST'])
def start_blind_assistant():
    if not blind_assistant.is_running():
        blind_assistant.start()
        return jsonify({'message': 'Blind Assistant started successfully.'}), 200
    else:
        return jsonify({'message': 'Blind Assistant is already running.'}), 400

# Route to stop the BlindAssistant
@app.route('/stop', methods=['POST'])
def stop_blind_assistant():
    if blind_assistant.is_running():
        blind_assistant.stop()
        return jsonify({'message': 'Blind Assistant stopped successfully.'}), 200
    else:
        return jsonify({'message': 'Blind Assistant is not running.'}), 400

# Route to get the state of the BlindAssistant
@app.route('/state', methods=['GET'])
def get_blind_assistant_state():
    state = {
        'running': blind_assistant.is_running()
    }
    return jsonify(state), 200

# Route to analyze an image
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    analysis_result = blind_assistant.analyze_image(image_file)
    return jsonify(analysis_result), 200

if __name__ == '__main__':
    app.run(debug=True)
