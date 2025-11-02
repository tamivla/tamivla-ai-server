/* Специфичные JavaScript функции для квантования */

// Загрузка анализа GPU
async function loadGPUAnalysis() {
    try {
        const data = await apiCall('/quantization/gpu-info');
        
        if (!data.available) {
            document.getElementById('gpu-stats').innerHTML = `
                <div class="stat-card warning">
                    <h3>Статус GPU</h3>
                    <div class="stat-value">❌ Не доступен</div>
                    <div>${data.error || 'GPU не обнаружен'}</div>
                </div>
            `;
            return;
        }
        
        let gpuStatsHtml = '';
        let gpuDetailsHtml = '<h3>Детальная информация:</h3>';
        
        for (const [gpuId, gpuInfo] of Object.entries(data.gpus)) {
            const freePercent = ((gpuInfo.free_gb / gpuInfo.total_gb) * 100).toFixed(1);
            
            gpuStatsHtml += `
                <div class="stat-card gpu-info">
                    <h3>${gpuInfo.name}</h3>
                    <div class="stat-value">${gpuInfo.free_gb} GB свободно</div>
                    <div>Всего: ${gpuInfo.total_gb} GB</div>
                    <div class="progress-bar">
                        <div class="progress-fill ${freePercent < 20 ? 'danger' : freePercent < 40 ? 'warning' : ''}" 
                             style="width: ${freePercent}%"></div>
                    </div>
                    <div>Свободно: ${freePercent}%</div>
                </div>
            `;
            
            gpuDetailsHtml += `
                <div style="margin-bottom: 15px;">
                    <strong>${gpuId}: ${gpuInfo.name}</strong><br>
                    • Всего памяти: ${gpuInfo.total_gb} GB<br>
                    • Занято: ${gpuInfo.allocated_gb} GB<br>
                    • Свободно: ${gpuInfo.free_gb} GB (${freePercent}%)<br>
                    • Compute: ${gpuInfo.compute_capability}
                </div>
            `;
        }
        
        document.getElementById('gpu-stats').innerHTML = gpuStatsHtml;
        document.getElementById('gpu-details').innerHTML = gpuDetailsHtml;
        
    } catch (error) {
        document.getElementById('gpu-stats').innerHTML = '<div class="error">Ошибка загрузки информации о GPU</div>';
    }
}

// Анализ выбранной модели
async function analyzeSelectedModel() {
    const modelSelect = document.getElementById('model-select');
    const selectedValue = modelSelect.value;
    
    if (selectedValue === 'custom') {
        document.getElementById('custom-model-group').style.display = 'block';
        return;
    } else if (selectedValue && selectedValue !== 'custom') {
        await analyzeModel(selectedValue);
    }
}

// Анализ кастомной модели
async function analyzeCustomModel() {
    const customModel = document.getElementById('custom-model').value.trim();
    if (customModel) {
        await analyzeModel(customModel);
    } else {
        showNotification('Введите имя модели', 'error');
    }
}

// Основная функция анализа модели
async function analyzeModel(modelName) {
    try {
        document.getElementById('model-analysis-results').innerHTML = '<div class="loading">Анализ модели...</div>';
        
        const data = await apiCall('/quantization/analyze', {
            method: 'POST',
            body: JSON.stringify({ model_name: modelName })
        });
        
        let recommendationsHtml = '';
        data.recommendations.forEach(rec => {
            const isRecommended = rec.level === data.best_recommendation.level;
            recommendationsHtml += `
                <div class="quant-option ${isRecommended ? 'recommended' : !rec.can_fit ? 'not-recommended' : ''}">
                    <div class="quant-level">${rec.level} (${rec.bits}-bit)</div>
                    <div class="quant-size">~${rec.estimated_size_gb} GB</div>
                    <div class="quant-status ${rec.can_fit ? 'status-ok' : 'status-error'}">
                        ${rec.can_fit ? '✅ Влезает' : '❌ Не влезает'}
                    </div>
                    <div>Качество: ${rec.quality}</div>
                    ${isRecommended ? '<div><strong>⭐ Рекомендуется</strong></div>' : ''}
                </div>
            `;
        });
        
        let suggestionsHtml = '<ul>';
        data.suggestions.forEach(suggestion => {
            suggestionsHtml += `<li>${suggestion}</li>`;
        });
        suggestionsHtml += '</ul>';
        
        const analysisHtml = `
            <div class="model-analysis-card">
                <div class="model-header">
                    <div class="model-name">${data.model_name}</div>
                    <div class="model-size">💾 Расчетный размер: ${data.estimated_size_gb} GB</div>
                </div>
                
                <div class="suggestions-list">
                    ${suggestionsHtml}
                </div>
                
                <h3>Варианты квантования:</h3>
                <div class="recommendation-grid">
                    ${recommendationsHtml}
                </div>
                
                <div style="margin-top: 20px;">
                    <button class="btn btn-success" onclick="quantizeModel('${data.model_name}', '${data.best_recommendation.level}')">
                        🚀 Квантовать как ${data.best_recommendation.level}
                    </button>
                    <button class="btn btn-primary" onclick="showQuantizationOptions('${data.model_name}')">
                        🔧 Другие варианты
                    </button>
                </div>
            </div>
        `;
        
        document.getElementById('model-analysis-results').innerHTML = analysisHtml;
        
    } catch (error) {
        document.getElementById('model-analysis-results').innerHTML = '<div class="error">Ошибка анализа модели</div>';
    }
}

// Пакетный анализ популярных моделей
async function analyzePopularModels() {
    try {
        document.getElementById('batch-analysis-results').innerHTML = '<div class="loading">Анализ популярных моделей...</div>';
        
        const data = await apiCall('/quantization/recommendations/popular-models');
        
        let resultsHtml = '<div class="stats-grid">';
        
        for (const [modelName, result] of Object.entries(data.popular_models)) {
            if (result.error) continue;
            
            const status = result.can_load ? '✅' : '❌';
            const colorClass = result.can_load ? 'success' : 'warning';
            
            resultsHtml += `
                <div class="stat-card ${colorClass}">
                    <h3>${modelName.split('/').pop()}</h3>
                    <div class="stat-value">${status}</div>
                    <div>Размер: ${result.estimated_size_gb} GB</div>
                    <div>Рекомендация: ${result.best_recommendation.level}</div>
                </div>
            `;
        }
        
        resultsHtml += '</div>';
        document.getElementById('batch-analysis-results').innerHTML = resultsHtml;
        
    } catch (error) {
        document.getElementById('batch-analysis-results').innerHTML = '<div class="error">Ошибка пакетного анализа</div>';
    }
}

// Запуск квантования
async function startQuantization() {
    const modelName = document.getElementById('quantize-model').value.trim();
    const quantLevel = document.getElementById('quantize-level').value;
    
    if (!modelName) {
        showNotification('Введите имя модели', 'error');
        return;
    }
    
    try {
        document.getElementById('quantization-status').innerHTML = '<div class="loading">Запуск квантования...</div>';
        
        const result = await apiCall(`/quantization/model/${encodeURIComponent(modelName)}/quantize?quantization_level=${quantLevel}`, {
            method: 'POST'
        });
        
        if (result.status === 'already_exists') {
            document.getElementById('quantization-status').innerHTML = `
                <div class="success">
                    ${result.message}
                </div>
            `;
        } else if (result.status === 'completed') {
            document.getElementById('quantization-status').innerHTML = `
                <div class="success">
                    ✅ ${result.message}<br>
                    Путь: ${result.quantized_path}
                </div>
            `;
        } else {
            document.getElementById('quantization-status').innerHTML = `
                <div class="info">
                    ⏳ ${result.message}
                </div>
            `;
        }
        
        showNotification(result.message, 'success');
        
    } catch (error) {
        document.getElementById('quantization-status').innerHTML = '<div class="error">Ошибка запуска квантования</div>';
        showNotification('Ошибка запуска квантования', 'error');
    }
}

// Быстрое квантование модели
async function quantizeModel(modelName, quantLevel) {
    try {
        showNotification(`Запуск квантования ${modelName}...`, 'info');
        
        const result = await apiCall(`/quantization/model/${encodeURIComponent(modelName)}/quantize?quantization_level=${quantLevel}`, {
            method: 'POST'
        });
        
        if (result.status === 'completed') {
            showNotification(`✅ ${result.message}`, 'success');
        } else if (result.status === 'already_exists') {
            showNotification(`ℹ️ ${result.message}`, 'info');
        } else {
            showNotification(`⏳ ${result.message}`, 'info');
        }
        
    } catch (error) {
        console.error('Quantization error:', error);
        showNotification(`❌ Ошибка квантования: ${error.message}`, 'error');
    }
}

// Показать варианты квантования
async function showQuantizationOptions(modelName) {
    try {
        const data = await apiCall(`/quantization/model/${encodeURIComponent(modelName)}/quantization-options`);
        
        let optionsHtml = '<h3>Все варианты квантования:</h3><div class="recommendation-grid">';
        
        data.quantization_options.forEach(option => {
            optionsHtml += `
                <div class="quant-option ${option.recommended ? 'recommended' : ''}">
                    <div class="quant-level">${option.level} (${option.bits}-bit)</div>
                    <div class="quant-size">~${option.estimated_size_gb} GB</div>
                    <div class="quant-status ${option.can_fit ? 'status-ok' : 'status-error'}">
                        ${option.can_fit ? '✅ Влезает' : '❌ Не влезает'}
                    </div>
                    <div>Качество: ${option.quality}</div>
                    <button class="btn btn-primary" onclick="quantizeModel('${modelName}', '${option.level}')" 
                            ${!option.can_fit ? 'disabled' : ''}>
                        Квантовать
                    </button>
                </div>
            `;
        });
        
        optionsHtml += '</div>';
        
        document.getElementById('model-analysis-results').innerHTML += `
            <div class="model-analysis-card">
                ${optionsHtml}
            </div>
        `;
        
    } catch (error) {
        showNotification('Ошибка загрузки вариантов квантования', 'error');
    }
}

// Загрузка квантованных моделей
async function loadQuantizedModels() {
    try {
        const data = await apiCall('/quantization/quantized-models');
        
        if (data.total_quantized === 0) {
            document.getElementById('quantized-models-list').innerHTML = '<p>Нет квантованных моделей</p>';
            return;
        }
        
        let modelsHtml = '<div class="stats-grid">';
        
        data.quantized_models.forEach(model => {
            modelsHtml += `
                <div class="stat-card success">
                    <h3>${model.original_name}</h3>
                    <div class="stat-value">${model.quantization_level}</div>
                    <div>Размер: ${model.size_mb ? formatSize(model.size_mb) : 'N/A'}</div>
                    <div>Путь: ${model.path.split('/').pop()}</div>
                </div>
            `;
        });
        
        modelsHtml += '</div>';
        document.getElementById('quantized-models-list').innerHTML = modelsHtml;
        
    } catch (error) {
        document.getElementById('quantized-models-list').innerHTML = '<div class="error">Ошибка загрузки списка</div>';
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Обработка выбора custom модели
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', function() {
            if (this.value === 'custom') {
                document.getElementById('custom-model-group').style.display = 'block';
            } else {
                document.getElementById('custom-model-group').style.display = 'none';
            }
        });
    }
    
    // Загрузка данных для активной вкладки
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) {
        if (activeTab.id === 'gpu-analysis') {
            loadGPUAnalysis();
        } else if (activeTab.id === 'quantized-models') {
            loadQuantizedModels();
        }
    }
});