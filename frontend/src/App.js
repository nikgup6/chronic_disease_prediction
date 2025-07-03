/* global __app_id, __firebase_config, __initial_auth_token */

import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, doc, setDoc, onSnapshot } from 'firebase/firestore';

// Define global variables for Firebase configuration provided by the environment
// For LOCAL DEVELOPMENT, hardcode your Firebase config here.
// REMEMBER TO REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL CONFIG FROM THE FIREBASE CONSOLE!
const LOCAL_FIREBASE_CONFIG = {
  apiKey: "AIzaSyDRZgG5mZ6xNSJ-7u8aLJXDYLbcW0Cvz_E",
  authDomain: "chronic-700a4.firebaseapp.com",
  projectId: "chronic-700a4",
  storageBucket: "chronic-700a4.firebasestorage.app",
  messagingSenderId: "957474842908",
  appId: "1:957474842908:web:777334f945715a461c72d9",
  measurementId: "G-33P53WWWX5"
};

// For LOCAL DEVELOPMENT, you might also want to set a default app ID if you're not using __app_id
const LOCAL_APP_ID = "chronic_app_id_test"; // Choose a unique ID for your local testing

// Use the provided global variables if they exist (in Canvas environment), otherwise use local ones
const appId = typeof __app_id !== 'undefined' ? __app_id : LOCAL_APP_ID;
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : LOCAL_FIREBASE_CONFIG;
const initialAuthToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;


// Helper function to generate a random UUID for anonymous users
const generateUUID = () => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

// Main App Component
const App = () => {
  const [db, setDb] = useState(null);
  const [auth, setAuth] = useState(null);
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    glucose: '',
    systolicBP: '',
    diastolicBP: '',
    triglycerides: '',
    hdl: '',
    waistCircumference: '',
    activityLevel: 'low',
    diet: 'unhealthy',
    sleep: 'poor',
    stress: 'high',
    alcohol: 'yes',
    smoking: 'yes',
    fatigue: 'yes',
  });
  const [risks, setRisks] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [message, setMessage] = useState('');
  const [isPredicting, setIsPredicting] = useState(false); // New loading state for prediction

  // Initialize Firebase and handle authentication
  useEffect(() => {
    const initFirebase = async () => {
      try {
        console.log("Initializing Firebase with config:", firebaseConfig); // Log config to verify
        const app = initializeApp(firebaseConfig);
        const firestore = getFirestore(app);
        const firebaseAuth = getAuth(app);
        setDb(firestore);
        setAuth(firebaseAuth);

        // Authenticate user
        if (initialAuthToken) {
          await signInWithCustomToken(firebaseAuth, initialAuthToken);
        } else {
          // IMPORTANT: For signInAnonymously to work, you MUST enable "Anonymous"
          // as a Sign-in method in your Firebase project's Authentication section.
          // Go to Firebase Console -> Authentication -> Sign-in method -> Anonymous.
          await signInAnonymously(firebaseAuth);
        }

        // Listen for auth state changes to get userId
        onAuthStateChanged(firebaseAuth, (user) => {
          if (user) {
            setUserId(user.uid);
            console.log("Authenticated user ID:", user.uid);
          } else {
            // Fallback for anonymous user if token not provided
            const anonId = generateUUID();
            setUserId(anonId);
            console.log("Anonymous user ID:", anonId);
          }
          setLoading(false);
        });

      } catch (err) {
        console.error("Firebase initialization or authentication error:", err);
        // Provide more specific guidance for common Firebase Auth errors
        let errorMessage = "Failed to initialize the application. Please try again later. Check console for details.";
        if (err.code === 'auth/configuration-not-found' || err.code === 'auth/invalid-api-key') {
          errorMessage = "Firebase configuration error: Please double-check your 'LOCAL_FIREBASE_CONFIG' values in App.js against your Firebase project settings, and ensure Firebase Authentication is enabled for your project.";
        } else if (err.code === 'auth/unauthorized-domain') {
          errorMessage = "Firebase authorization error: Your domain (localhost) might not be authorized. Add 'localhost' to authorized domains in Firebase Console -> Authentication -> Settings -> Authorized domains.";
        } else if (err.code === 'auth/operation-not-allowed') {
          errorMessage = "Firebase authentication method not enabled: Please enable 'Anonymous' sign-in method in Firebase Console -> Authentication -> Sign-in method.";
        }
        setError(errorMessage);
        setLoading(false);
      }
    };

    initFirebase();
  }, []);

  // Fetch user data if userId is available
  useEffect(() => {
    if (db && userId) {
      const userDocRef = doc(db, `artifacts/${appId}/users/${userId}/health_data`, 'current');
      const unsubscribe = onSnapshot(userDocRef, (docSnap) => {
        if (docSnap.exists()) {
          const data = docSnap.data();
          setFormData(data.formData || formData);
          setRisks(data.risks || null);
          setRecommendations(data.recommendations || []);
          setMessage('Data loaded successfully.');
        } else {
          setMessage('No previous data found. Please enter your health information.');
        }
      }, (err) => {
        console.error("Error fetching document:", err);
        setError("Failed to load your health data.");
      });

      return () => unsubscribe(); // Cleanup snapshot listener
    }
  }, [db, userId]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!db || !userId) {
      setError("System not ready. Please wait for initialization.");
      return;
    }
    setIsPredicting(true);
    setMessage('Sending data to ML model for prediction...');
    setError(null); // Clear previous errors

    try {
      const parsedData = {
        glucose: parseFloat(formData.glucose),
        systolicBP: parseFloat(formData.systolicBP),
        diastolicBP: parseFloat(formData.diastolicBP),
        triglycerides: parseFloat(formData.triglycerides),
        hdl: parseFloat(formData.hdl),
        waistCircumference: parseFloat(formData.waistCircumference),
        activityLevel: formData.activityLevel,
        diet: formData.diet,
        sleep: formData.sleep,
        stress: formData.stress,
        alcohol: formData.alcohol,
        smoking: formData.smoking,
        fatigue: formData.fatigue,
      };

      // Send data to Flask backend for ML prediction
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parsedData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get predictions from backend.');
      }

      const predictedRisks = await response.json();
      setRisks(predictedRisks);

      // Generate recommendations based on predicted risks
      const generatedRecommendations = getRecommendations(predictedRisks, parsedData);
      setRecommendations(generatedRecommendations);

      // Save data to Firestore
      const userDocRef = doc(db, `artifacts/${appId}/users/${userId}/health_data`, 'current');
      await setDoc(userDocRef, {
        formData: parsedData,
        risks: predictedRisks,
        recommendations: generatedRecommendations,
        timestamp: new Date().toISOString(),
      });

      setMessage('Health data, risks, and recommendations updated successfully!');
    } catch (err) {
      console.error("Error during prediction or saving data:", err);
      setError(`Error: ${err.message}. Please ensure the backend server is running.`);
      setMessage(''); // Clear successful message if error occurs
    } finally {
      setIsPredicting(false);
    }
  };

  // --- Personalized Intervention Planner (Rule-Based) ---
  // This logic remains on the frontend, using the ML-predicted risks
  const getRecommendations = (risks, data) => {
    const recs = new Set(); // Use a Set to avoid duplicate recommendations

    if (risks.preDiabetes === 'medium' || risks.preDiabetes === 'high') {
      recs.add('Reduce sugar and refined carbohydrates intake.');
      recs.add('Increase fiber-rich foods (whole grains, vegetables, fruits).');
      recs.add('Aim for at least 30 minutes of moderate-intensity exercise most days of the week.');
    }
    if (risks.hypertension === 'high' || risks.hypertension === 'medium') { // Changed from 'elevated', 'stage 1', 'stage 2' to match ML output
      recs.add('Reduce sodium intake (limit processed foods, salty snacks).');
      recs.add('Follow the DASH (Dietary Approaches to Stop Hypertension) diet.');
      recs.add('Engage in regular aerobic exercise like brisk walking or cycling.');
      recs.add('Manage stress through techniques like meditation or yoga.');
    }
    if (risks.metabolicSyndrome === 'medium' || risks.metabolicSyndrome === 'high') {
      recs.add('Focus on weight management, especially reducing belly fat.');
      recs.add('Increase physical activity to at least 150 minutes per week.');
      recs.add('Limit saturated and trans fats; choose healthy fats (avocado, nuts, olive oil).');
      recs.add('Consider consulting a nutritionist for a personalized meal plan.');
    }

    // Lifestyle-based recommendations regardless of specific disease risk
    if (data.activityLevel === 'low') {
      recs.add('Incorporate more physical activity into your daily routine (e.g., stairs instead of elevator).');
    }
    if (data.diet === 'unhealthy') {
      recs.add('Prioritize whole, unprocessed foods and reduce fast food consumption.');
    }
    if (data.sleep === 'poor') {
      recs.add('Establish a consistent sleep schedule and create a relaxing bedtime routine.');
    }
    if (data.stress === 'high') {
      recs.add('Practice stress-reduction techniques such as deep breathing or mindfulness.');
    }
    if (data.alcohol === 'yes') {
      recs.add('Moderate alcohol consumption or consider reducing it.');
    }
    if (data.smoking === 'yes') {
      recs.add('Seek support to quit smoking for significant health benefits.');
    }

    return Array.from(recs); // Convert Set back to Array
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-xl font-semibold text-gray-700">Loading application...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 sm:p-8 font-inter text-gray-800">
      <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden">
        <header className="bg-indigo-600 text-white p-6 rounded-t-3xl">
          <h1 className="text-3xl font-bold text-center mb-2">Health Early Warning System</h1>
          <p className="text-center text-indigo-200">Monitor, Predict, Prevent. Your personalized health guardian.</p>
          {userId && (
            <p className="text-center text-sm mt-2 text-indigo-300">
              User ID: <span className="font-mono bg-indigo-700 px-2 py-1 rounded-md text-xs">{userId}</span>
            </p>
          )}
        </header>

        <main className="p-6 sm:p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Data Input Form */}
          <section className="bg-gray-50 p-6 rounded-2xl shadow-inner">
            <h2 className="text-2xl font-semibold text-indigo-700 mb-6 border-b pb-3 border-indigo-200">
              <i className="lucide lucide-clipboard-list inline-block mr-2"></i>
              Enter Your Health Data
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Biomarkers */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="glucose" className="block text-sm font-medium text-gray-700 mb-1">Fasting Glucose (mg/dL)</label>
                  <input
                    type="number"
                    id="glucose"
                    name="glucose"
                    value={formData.glucose}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 95"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="systolicBP" className="block text-sm font-medium text-gray-700 mb-1">Systolic BP (mmHg)</label>
                  <input
                    type="number"
                    id="systolicBP"
                    name="systolicBP"
                    value={formData.systolicBP}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 120"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="diastolicBP" className="block text-sm font-medium text-gray-700 mb-1">Diastolic BP (mmHg)</label>
                  <input
                    type="number"
                    id="diastolicBP"
                    name="diastolicBP"
                    value={formData.diastolicBP}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 80"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="triglycerides" className="block text-sm font-medium text-gray-700 mb-1">Triglycerides (mg/dL)</label>
                  <input
                    type="number"
                    id="triglycerides"
                    name="triglycerides"
                    value={formData.triglycerides}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 140"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="hdl" className="block text-sm font-medium text-gray-700 mb-1">HDL Cholesterol (mg/dL)</label>
                  <input
                    type="number"
                    id="hdl"
                    name="hdl"
                    value={formData.hdl}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 50"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="waistCircumference" className="block text-sm font-medium text-gray-700 mb-1">Waist Circumference (cm)</label>
                  <input
                    type="number"
                    id="waistCircumference"
                    name="waistCircumference"
                    value={formData.waistCircumference}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., 90"
                    required
                  />
                </div>
              </div>

              {/* Lifestyle Factors */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="activityLevel" className="block text-sm font-medium text-gray-700 mb-1">Activity Level</label>
                  <select
                    id="activityLevel"
                    name="activityLevel"
                    value={formData.activityLevel}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="low">Low (Sedentary)</option>
                    <option value="moderate">Moderate (Some exercise)</option>
                    <option value="high">High (Active daily)</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="diet" className="block text-sm font-medium text-gray-700 mb-1">Diet Quality</label>
                  <select
                    id="diet"
                    name="diet"
                    value={formData.diet}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="unhealthy">Unhealthy (Processed, sugary)</option>
                    <option value="moderate">Moderate (Balanced but inconsistent)</option>
                    <option value="healthy">Healthy (Whole foods, balanced)</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="sleep" className="block text-sm font-medium text-gray-700 mb-1">Sleep Quality</label>
                  <select
                    id="sleep"
                    name="sleep"
                    value={formData.sleep}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="poor">Poor (Less than 6h, interrupted)</option>
                    <option value="average">Average (6-7h, some interruptions)</option>
                    <option value="good">Good (7-9h, consistent)</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="stress" className="block text-sm font-medium text-gray-700 mb-1">Stress Level</label>
                  <select
                    id="stress"
                    name="stress"
                    value={formData.stress}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="high">High</option>
                    <option value="moderate">Moderate</option>
                    <option value="low">Low</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="alcohol" className="block text-sm font-medium text-gray-700 mb-1">Alcohol Consumption</label>
                  <select
                    id="alcohol"
                    name="alcohol"
                    value={formData.alcohol}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="smoking" className="block text-sm font-medium text-gray-700 mb-1">Smoking Status</label>
                  <select
                    id="smoking"
                    name="smoking"
                    value={formData.smoking}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
              </div>

              {/* Symptoms */}
              <div>
                <label htmlFor="fatigue" className="block text-sm font-medium text-gray-700 mb-1">Frequent Fatigue?</label>
                <select
                  id="fatigue"
                  name="fatigue"
                  value={formData.fatigue}
                  onChange={handleChange}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>

              <button
                type="submit"
                className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition duration-200 ease-in-out shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isPredicting}
              >
                {isPredicting ? 'Calculating...' : 'Get My Health Report'}
              </button>
            </form>
            {message && (
              <p className={`mt-4 text-center text-sm ${error ? 'text-red-600' : 'text-green-600'}`}>
                {message}
              </p>
            )}
            {error && (
              <p className="mt-4 text-center text-sm text-red-600">
                {error}
              </p>
            )}
          </section>

          {/* Risk Prediction & Recommendations */}
          <section className="bg-white p-6 rounded-2xl shadow-lg">
            <h2 className="text-2xl font-semibold text-indigo-700 mb-6 border-b pb-3 border-indigo-200">
              <i className="lucide lucide-activity inline-block mr-2"></i>
              Your Health Insights
            </h2>

            {risks && (
              <div className="mb-6">
                <h3 className="text-xl font-medium text-gray-900 mb-3 flex items-center">
                  <i className="lucide lucide-alert-triangle text-yellow-500 mr-2"></i>
                  Predicted Risks:
                </h3>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-center">
                        <span className="font-semibold w-36">Pre-diabetes:</span>
                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                          risks.preDiabetes === 'high' ? 'bg-red-100 text-red-800' :
                          risks.preDiabetes === 'medium' ? 'bg-orange-100 text-orange-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {risks.preDiabetes.toUpperCase()}
                        </span>
                      </li>
                      <li className="flex items-center">
                        <span className="font-semibold w-36">Hypertension:</span>
                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                          risks.hypertension === 'high' ? 'bg-red-100 text-red-800' :
                          risks.hypertension === 'medium' ? 'bg-orange-100 text-orange-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {risks.hypertension.toUpperCase()}
                        </span>
                      </li>
                      <li className="flex items-center">
                        <span className="font-semibold w-36">Metabolic Syndrome:</span>
                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                          risks.metabolicSyndrome === 'high' ? 'bg-red-100 text-red-800' :
                          risks.metabolicSyndrome === 'medium' ? 'bg-orange-100 text-orange-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {risks.metabolicSyndrome.toUpperCase()}
                        </span>
                      </li>
                    </ul>
                  </div>
                )}

                {recommendations.length > 0 && (
                  <div>
                    <h3 className="text-xl font-medium text-gray-900 mb-3 flex items-center">
                      <i className="lucide lucide-lightbulb text-blue-500 mr-2"></i>
                      Personalized Recommendations:
                    </h3>
                    <ul className="list-disc pl-5 space-y-2 text-gray-700">
                      {recommendations.map((rec, index) => (
                        <li key={index} className="bg-blue-50 p-3 rounded-lg border border-blue-100 shadow-sm text-sm">
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {!risks && !recommendations.length && (
                  <div className="text-center text-gray-500 py-10">
                    <p className="mb-2">Enter your health data in the form to get your personalized insights.</p>
                    <i className="lucide lucide-arrow-left text-gray-400 text-4xl"></i>
                  </div>
                )}
              </section>
            </main>

            <footer className="bg-indigo-600 text-white text-center p-4 rounded-b-3xl text-sm">
              &copy; {new Date().getFullYear()} Health Early Warning System. All rights reserved.
            </footer>
          </div>
        </div>
      );
    };

    export default App;
