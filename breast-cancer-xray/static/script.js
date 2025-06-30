document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const fileName = document.getElementById('fileName');
    const previewImage = document.getElementById('previewImage');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const resultContainer = document.getElementById('result');
    
    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(file);
            
            // Enable analyze button
            analyzeButton.disabled = false;
        } else {
            fileName.textContent = 'Choose an image...';
            imagePreview.style.display = 'none';
            analyzeButton.disabled = true;
        }
    });
    
    analyzeButton.addEventListener('click', function() {
        const file = imageUpload.files[0];
        if (!file) return;
        
        // Show loading state
        resultContainer.innerHTML = '<p class="loading">Analyzing image, please wait...</p>';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Analysis failed') });
            }
            return response.json();
        })
        .then(data => {
            const confidencePercent = (data.confidence * 100).toFixed(2);
            let recommendation = "";
            
            if (data.class_name === 'Malignant') {
                recommendation = "Please consult with a healthcare professional immediately.";
            } else if (data.class_name === 'Benign') {
                recommendation = "Regular check-ups are recommended.";
            } else {
                recommendation = "No abnormalities detected. Maintain regular screening schedule.";
            }
            
            resultContainer.innerHTML = `
                <p>Diagnosis: <span class="${data.class_name.toLowerCase()}">${data.class_name}</span></p>
                <p>Confidence: ${confidencePercent}%</p>
                <div class="recommendation">
                    <p><strong>Recommendation:</strong> ${recommendation}</p>
                </div>
                <details class="debug-info">
                    <summary>Detailed probabilities</summary>
                    <p>Benign: ${(data.all_predictions.Benign * 100).toFixed(2)}%</p>
                    <p>Malignant: ${(data.all_predictions.Malignant * 100).toFixed(2)}%</p>
                    <p>Normal: ${(data.all_predictions.Normal * 100).toFixed(2)}%</p>
                </details>
            `;
        })
        .catch(error => {
            resultContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        });
    });
});