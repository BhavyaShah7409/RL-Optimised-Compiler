from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from .compiler import SourceToSourceCompiler

app = Flask(__name__)
CORS(app, origins=["*"])

# Initialize compiler
compiler = SourceToSourceCompiler()

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'RL-Optimized Source to Source Compiler API is running'
    })

@app.route('/api/compile', methods=['POST'])
def compile_code():
    try:
        data = request.get_json()
        source_code = data.get('source_code', '')
        input_data = data.get('input_data', '')
        
        if not source_code.strip():
            return jsonify({
                'success': False,
                'error': 'No source code provided'
            })
        
        # Compile and execute
        result = compiler.compile_and_execute(source_code, input_data)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
