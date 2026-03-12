const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('imageUpload');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const clearBoxBtn = document.getElementById('clearBoxBtn');
const processBtn = document.getElementById('processBtn');
const processLoader = document.getElementById('processLoader');
const btnText = document.querySelector('.btn-text');
const errorMsg = document.getElementById('errorMessage');

const emptyState = document.getElementById('emptyState');
const stepOriginal = document.getElementById('stepOriginal');
const stepPrompt = document.getElementById('stepPrompt');
const stepResult = document.getElementById('stepResult');

const resOriginal = document.getElementById('resOriginal');
const resPrompt = document.getElementById('resPrompt');
const resSegmented = document.getElementById('resSegmented');

let currentFile = null;
let currentImage = null;

// Drawing state
let isDrawing = false;
let startX = 0;
let startY = 0;
let boxCoords = null; // [x1, y1, x2, y2] relative to original image size

// API Config
const API_BASE = 'http://localhost:8000';

// Upload Handlers
uploadContainer.addEventListener('click', () => fileInput.click());

uploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', () => {
    uploadContainer.classList.remove('dragover');
});

uploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadContainer.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/') && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
        showError("Invalid image type: " + (file.type || "unknown") + ". Please use JPG or PNG.");
        return;
    }

    currentFile = file;
    errorMsg.style.display = 'none';

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            initCanvas();
            uploadContainer.style.display = 'none';
            canvasWrapper.style.display = 'flex';
        };
        img.onerror = () => {
            showError("Failed to decode image. The file might be corrupted or in an unsupported format.");
        };
        img.src = e.target.result;
    };
    reader.onerror = () => {
        showError("Failed to read the file from disk.");
    };
    
    try {
        reader.readAsDataURL(file);
    } catch (err) {
        showError("Error starting file read: " + err.message);
    }
}

function initCanvas() {
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;
    ctx.drawImage(currentImage, 0, 0);
    boxCoords = null;
    document.getElementById('resultsSection').style.display = 'none';
}

function redrawCanvas() {
    if (!currentImage) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    if (boxCoords) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = Math.max(2, canvas.width / 500); // Dynamic line width
        const width = boxCoords[2] - boxCoords[0];
        const height = boxCoords[3] - boxCoords[1];
        ctx.strokeRect(boxCoords[0], boxCoords[1], width, height);
    }
}

// Mouse Handlers to Draw Box
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    // Calculate mapping between CSS pixels and intrinsic Canvas pixels
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

canvas.addEventListener('mousedown', (e) => {
    if (!currentImage) return;
    const { x, y } = getCanvasCoords(e);
    isDrawing = true;
    startX = x;
    startY = y;
    boxCoords = null; // reset box on new click
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const { x, y } = getCanvasCoords(e);

    redrawCanvas();
    // Draw temporary rectangle
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = Math.max(2, canvas.width / 500);
    ctx.strokeRect(startX, startY, x - startX, y - startY);
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    const { x, y } = getCanvasCoords(e);

    // Ensure box is valid (has some width/height)
    if (Math.abs(x - startX) > 5 && Math.abs(y - startY) > 5) {
        boxCoords = [
            Math.min(startX, x),
            Math.min(startY, y),
            Math.max(startX, x),
            Math.max(startY, y)
        ];
    } else {
        boxCoords = null;
    }
    redrawCanvas();
});

// Controls
clearBoxBtn.addEventListener('click', () => {
    boxCoords = null;
    redrawCanvas();
});

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.style.display = 'block';
}

processBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Set UI to loading
    processBtn.disabled = true;
    btnText.textContent = "Processing...";
    processLoader.style.display = "block";
    errorMsg.style.display = 'none';

    // Prepare form data
    const formData = new FormData();
    formData.append('image', currentFile);

    if (boxCoords) {
        // SAM expects [x1, y1, x2, y2] array
        formData.append('box', JSON.stringify(boxCoords));
    }

    try {
        const req = await fetch(`${API_BASE}/api/segment`, {
            method: 'POST',
            body: formData
        });

        const res = await req.json();

        if (!req.ok) {
            throw new Error(res.error || `Server error: ${req.status}`);
        }

        // Success: Update results UI
        document.getElementById('resultsSection').style.display = 'flex';

        resOriginal.src = `${API_BASE}${res.original_url}`;
        resSegmented.src = `${API_BASE}${res.segmented_url}`;

        // Smooth scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        showError(err.message);
    } finally {
        // Reset UI
        processBtn.disabled = false;
        btnText.textContent = "Generate Segmentation";
        processLoader.style.display = "none";
    }
});
