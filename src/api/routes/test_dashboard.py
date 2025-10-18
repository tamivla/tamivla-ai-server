"""
Test dashboard for diagnostics
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/test-dashboard", response_class=HTMLResponse)
async def test_dashboard():
    """
    Simple test page
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Dashboard</title>
    </head>
    <body>
        <h1>✅ Test Dashboard Works!</h1>
        <p>If you see this text, basic HTML works.</p>
        <p>Server time: <span id="time">loading...</span></p>
        <button onclick="loadData()">Test API</button>
        <div id="result"></div>
        
        <script>
            document.getElementById('time').textContent = new Date().toLocaleString();
            
            async function loadData() {
                try {
                    const response = await fetch('/system/status');
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 
                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '❌ API Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)