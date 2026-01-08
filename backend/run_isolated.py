import http.server
import socketserver
import json
import os
from datetime import datetime

class IsolatedAPIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health' or self.path == '/api/v1/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "status": "healthy",
                "message": "Isolated API server running",
                "version": "1.0.0",
                "environment": "isolated_network",
                "timestamp": datetime.now().isoformat(),
                "features": [
                    "Basic API routing",
                    "Health check endpoints", 
                    "CORS support",
                    "Static file serving"
                ]
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Liver Segmentation API - Isolated Mode</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; }
.header { background: #f0f0f0; padding: 20px; border-radius: 8px; }
.status { color: green; font-weight: bold; }
.warning { background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }
</style>
</head>
<body>
<div class="header">
    <h1>🏥 Liver Segmentation API - Isolated Mode</h1>
    <p class="status">✅ Status: Running</p>
</div>
<div class="warning">
    <strong>⚠️ Внимание:</strong> Сервер работает в изолированном режиме без внешних зависимостей.
</div>
<p><a href="/health">🔍 Health Check</a></p>
<p><strong>Note:</strong> Running in isolated mode without external dependencies</p>
</body></html>'''
            self.wfile.write(html.encode())
    
    def do_POST(self):
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "error": "POST not available in isolated mode",
            "message": "This endpoint requires external dependencies (FastAPI, DICOM libs, etc.)",
            "status": 404
        }
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        # Упрощенное логирование
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

def run_server():
    port = int(os.environ.get('PORT', 8000))
    print("=" * 50)
    print("🏥 LIVER SEGMENTATION API - ISOLATED MODE")
    print("=" * 50)
    print(f"🌐 Server running on port {port}")
    print(f"📖 Health check: http://localhost:{port}/health")
    print("🔒 Environment: Isolated Network")
    print("=" * 50)
    
    with socketserver.TCPServer(("", port), IsolatedAPIHandler) as httpd:
        print("✅ Server started successfully!")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    run_server()