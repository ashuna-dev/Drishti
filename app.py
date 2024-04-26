from flask import Flask, render_template, Response, redirect, url_for,jsonify
from blindenv import BlindAssistant
import sys
import textwrap
import time

app = Flask(__name__)
blind_assistant = None
analysis_text = ""  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_program', methods=['POST'])
def start_program():
    global blind_assistant
    if blind_assistant is None:
        blind_assistant = BlindAssistant()
    return redirect(url_for('assistant'))

@app.route('/end_program', methods=['POST'])
def end_program():
    global blind_assistant
    if blind_assistant is not None:
        blind_assistant.end_program()
        blind_assistant = None
    return 'Program ended', 200

@app.route('/assistant')
def assistant():
    return render_template('assistant.html')

@app.route('/get_analysis_text')
def get_analysis_text():
    global blind_assistant
    if blind_assistant is not None:
        time.sleep(15)
        analysis_text = blind_assistant.analysis_text
        wrapped_text = textwrap.fill(analysis_text, width=40)
        return f'Analysis Text={analysis_text}'
    else:
        return 'Analysis Text=Blind Assistant is not running'

@app.route('/video_feed')
def video_feed():
    if blind_assistant is not None:
        return Response(blind_assistant.generate_frames_with_audio(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Blind Assistant is not running", 200

if __name__ == '__main__':
    app.run(debug=True)
