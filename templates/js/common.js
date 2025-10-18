/* Общие JavaScript функции для всех дашбордов */

// Функции для переключения вкладок
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Универсальная функция для API запросов
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Функция для показа уведомлений
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div style="position: fixed; top: 20px; right: 20px; padding: 15px; border-radius: 5px; 
                    background: ${type === 'error' ? '#e74c3c' : type === 'success' ? '#27ae60' : '#3498db'}; 
                    color: white; z-index: 1000; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
            ${message}
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Функция для форматирования размера
function formatSize(mb) {
    if (mb < 1024) {
        return `${mb.toFixed(1)} MB`;
    } else {
        return `${(mb / 1024).toFixed(1)} GB`;
    }
}

// Функция для создания карточки модели
function createModelCard(model) {
    const sizeGB = (model.size_mb / 1024).toFixed(1);
    const displayName = model.display_name || model.name.replace('models--', '').replace(/--/g, '/');
    
    return `
        <div class="model-card">
            <div class="model-header">
                <div class="model-name">${displayName}</div>
                <div class="model-type">${model.model_type || 'unknown'}</div>
            </div>
            <div class="model-size">📦 ${sizeGB} GB</div>
            <div>🏗️ ${model.architecture || 'Неизвестно'}</div>
            <div style="margin-top: 15px;">
                <button class="btn btn-success" onclick="loadModel('${displayName}')">🔄 Загрузить</button>
                <button class="btn btn-warning" onclick="unloadModel('${displayName}')">🗑️ Выгрузить</button>
            </div>
        </div>
    `;
}