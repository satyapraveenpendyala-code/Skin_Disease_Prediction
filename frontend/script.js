// --- 1. IMAGE PREVIEW FUNCTION ---
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const output = document.getElementById('preview');
            output.src = e.target.result;
            output.style.display = 'block';
            
            // Hide the helper text once image is shown
            const uploadText = document.getElementById('uploadText');
            if (uploadText) {
                uploadText.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
    }
}

// --- 2. MAIN PREDICTION FUNCTION ---
async function predict() {
    // A. Get Inputs
    const fileInput = document.getElementById('fileInput');
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const localization = document.getElementById('localization').value;

    // B. Validation
    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }

    // C. Prepare Data for Backend
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('age', age);
    formData.append('gender', gender);
    formData.append('localization', localization);

    // D. Update Button State (Loading...)
    const btn = document.querySelector('.btn');
    const originalText = btn.innerText;
    btn.innerText = "Processing Analysis...";
    btn.disabled = true;
    btn.style.cursor = "wait";

    try {
        // E. Send Request to Flask API
        // Ensure this matches your Python running port (usually 5000)
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();

        if (data.diagnosis === "Invalid Image") {
            alert(data.error_message);
            btn.innerText = originalText;
            btn.disabled = false;
            btn.style.cursor = "pointer";
            return;
        }

        if (!response.ok) {
            throw new Error(data.error || "Backend server error");
        }

        // --- F. UI ANIMATION (Expand the Box) ---
        // This class triggers the CSS transition to widen the container
        const container = document.getElementById('mainContainer');
        container.classList.add('expanded');

        // --- G. UPDATE RESULTS ---
        const diagnosisElement = document.getElementById('diagnosis');
        const confidenceElement = document.getElementById('confidence');
        
        // Update Text
        diagnosisElement.innerText = data.diagnosis;
        confidenceElement.innerText = data.confidence;

        // --- H. INTELLIGENT COLOR CODING ---
        // Turn text RED if it is a dangerous cancer, BLUE/GREEN otherwise.
        const condition = data.diagnosis.toLowerCase();
        if (condition.includes('melanoma') || condition.includes('carcinoma')) {
            diagnosisElement.style.color = "#c0392b"; // Red (Danger)
        } else {
            diagnosisElement.style.color = "#27ae60"; // Green (Likely Benign)
        }

        // --- I. HANDLE HEATMAP IMAGE ---
        const heatmapImg = document.getElementById('heatmapOutput');
        if (data.heatmap_url) {
            // Add timestamp (?t=...) to force browser to reload the image
            // This prevents showing the 'old' heatmap from the previous prediction
            heatmapImg.src = data.heatmap_url + "?t=" + new Date().getTime();
            heatmapImg.style.display = 'block';
        } else {
            heatmapImg.style.display = 'none';
        }

    } catch (err) {
        console.error("Analysis Error:", err);
        alert("Failed to analyze image. Please check if the Python server is running.\n\nError: " + err.message);
    } finally {
        // J. Reset Button
        btn.innerText = originalText;
        btn.disabled = false;
        btn.style.cursor = "pointer";
    }
}