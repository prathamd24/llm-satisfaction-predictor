// ── Firebase Configuration ─────────────────────────────────────────────────────
const firebaseConfig = {
    projectId: "llm-satisfaction-predictor",
    appId: "1:780525983813:web:126bb8f7c9c04e61d0ab2c",
    storageBucket: "llm-satisfaction-predictor.firebasestorage.app",
    apiKey: "AIzaSyBje0aeIF3174WbY9mPDrSDQCbQd3Bc6WU",
    authDomain: "llm-satisfaction-predictor.firebaseapp.com",
    messagingSenderId: "780525983813",
    measurementId: "G-N6SWNTD3G7"
};

firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const provider = new firebase.auth.GoogleAuthProvider();

// ── Configuration ────────────────────────────────────────────────────────────
// The FastAPI backend URL
const BACKEND_URL = "https://llm-satisfaction-predictor.onrender.com"; 

// ── DOM Elements ──────────────────────────────────────────────────────────────
const authView = document.getElementById('auth-view');
const appView = document.getElementById('app-view');
const loginBtn = document.getElementById('google-login-btn');
const logoutBtn = document.getElementById('logout-btn');
const userName = document.getElementById('user-name');
const authError = document.getElementById('auth-error');

const predictForm = document.getElementById('predict-form');
const predictBtn = document.getElementById('predict-btn');
const resultContainer = document.getElementById('result-container');
const resultScore = document.getElementById('result-score');
const resultScale = document.getElementById('result-scale');
const resultInterpretation = document.getElementById('result-interpretation');
const apiError = document.getElementById('api-error');
const backendWarning = document.getElementById('backend-warning');

let currentIdToken = null;

// ── Authentication Flow ───────────────────────────────────────────────────────
auth.onAuthStateChanged(async (user) => {
    if (user) {
        userName.textContent = user.displayName || user.email;
        currentIdToken = await user.getIdToken();
        
        authView.classList.remove('active');
        appView.classList.add('active');
        
        // Asynchronously check backend
        fetch(`${BACKEND_URL}/health`).then(res => {
            if (!res.ok) backendWarning.style.display = 'block';
        }).catch(() => {
            backendWarning.style.display = 'block';
        });

    } else {
        currentIdToken = null;
        appView.classList.remove('active');
        authView.classList.add('active');
    }
});

loginBtn.addEventListener('click', () => {
    authError.textContent = "";
    loginBtn.disabled = true;
    
    auth.signInWithPopup(provider).catch((error) => {
        console.error("Auth Error:", error);
        
        // Specifically handle the CONFIGURATION_NOT_FOUND error with clear instructions
        if (error.message.includes("CONFIGURATION_NOT_FOUND")) {
            authError.innerHTML = "<b>Configuration Error:</b> Google Sign-In is not enabled.<br><br>Please go to your Firebase Console &rarr; Authentication &rarr; Sign-in method, and enable <b>Google</b>.";
        } else {
            authError.textContent = error.message;
        }
        loginBtn.disabled = false;
    });
});

logoutBtn.addEventListener('click', () => {
    auth.signOut();
    resultContainer.style.display = 'none';
});

// ── API Logic ────────────────────────────────────────────────────────────────
predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!currentIdToken) return;

    apiError.textContent = "";
    resultContainer.style.display = 'none';
    
    const originalText = predictBtn.textContent;
    predictBtn.disabled = true;
    predictBtn.textContent = "Processing...";

    const payload = {
        model_name: document.getElementById('model_name').value,
        application_domain: document.getElementById('application_domain').value,
        task_type: document.getElementById('task_type').value,
        rag_enabled: parseInt(document.getElementById('rag_enabled').value),
        prompt_length: parseInt(document.getElementById('prompt_length').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top_p').value)
    };

    try {
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${currentIdToken}`
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            displayResult(data.predicted_satisfaction, data.scale);
        } else if (response.status === 401) {
            apiError.textContent = "Session expired. Please sign out and sign in again.";
            auth.signOut();
        } else {
            try {
                const errData = await response.json();
                apiError.textContent = `Server Error ${response.status}: ` + (errData.detail || JSON.stringify(errData));
            } catch (e) {
                apiError.textContent = `Server error ${response.status}. Make sure the backend is running.`;
            }
        }
    } catch (error) {
        apiError.textContent = `Network error: Could not reach backend at ${BACKEND_URL}.`;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = originalText;
    }
});

function displayResult(score, scale) {
    resultContainer.style.display = 'block';
    resultScore.textContent = score.toFixed(2);
    
    // Clear old classes
    resultInterpretation.className = "result-footer";
    
    if (score >= 4.5) {
        resultInterpretation.textContent = "Very high expected satisfaction. The configuration is optimal.";
        resultInterpretation.classList.add("status-success");
    } else if (score >= 4.0) {
        resultInterpretation.textContent = "High expected satisfaction. Users generally report positive experiences.";
        resultInterpretation.classList.add("status-info");
    } else if (score >= 3.5) {
        resultInterpretation.textContent = "Moderate satisfaction. Consider refining the prompt or parameters.";
        resultInterpretation.classList.add("status-warning");
    } else {
        resultInterpretation.textContent = "Low satisfaction expected. This configuration may lead to poor responses.";
        resultInterpretation.classList.add("status-error");
    }
    
    // Smooth scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
