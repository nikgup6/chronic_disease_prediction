My Health Early Warning System
Hey there! So, this is a cool little project I've been working on. It's basically an early warning system for some common health stuff like pre-diabetes, high blood pressure, and metabolic syndrome. The idea is to help people catch these things early so they can do something about it.

What it Does
You put in some of your health data – like your blood sugar, blood pressure, and some things about your lifestyle (how much you exercise, what you eat, sleep, stress, all that). Then, the system tries to figure out if you're at risk for these chronic diseases. If it thinks you are, it gives you some personalized tips on what you can do, like changing your diet or getting more active.

It's built to be pretty smart, using some machine learning magic in the background, but I tried to make it simple for anyone to use.

How to Get It Running (Local Setup)
Okay, so if you wanna run this on your own computer, here's what you need to do. It's got two main parts: the "brain" (backend) and the "face" (frontend).

First, the Brain (Backend - Python)
Get the files: Make sure you have the backend folder and all the stuff inside it. Also, you'll need the NHANES dataset CSV files (demographics.csv, examination.csv, labs.csv, questionnaire.csv) right there in the backend folder. You can download this dataset from Kaggle.

Install Python stuff: Open your terminal (like Command Prompt on Windows or Terminal on Mac/Linux), go into the backend folder:

cd health-warning-system/backend

Then, install all the Python libraries it needs. It's a good idea to make a "virtual environment" first, keeps things tidy.

python -m venv venv

# On Windows: venv\Scripts\activate

# On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt

Train the smart part: This is where the machine learning models learn from the health data. It takes a bit.

python app.py train

You should see some messages about models being saved. If it complains about files, double check those CSVs are there.

Start the brain: Once training is done, keep this terminal open and run the backend server:

python app.py

It should say something like "Running on http://127.0.0.1:5000".

Next, the Face (Frontend - React)
Go to the frontend folder: Open a new terminal window and go into the frontend folder:

cd health-warning-system/frontend

Install Node.js stuff:

npm install

Little Firebase thing: This app uses Firebase for saving your data. You'll need to go to the Firebase website (console.firebase.google.com), make a project, and get your project's config details (it's a little piece of code with apiKey, projectId etc.). You need to paste these into frontend/src/App.js where it says LOCAL_FIREBASE_CONFIG. Also, make sure you enable "Anonymous" sign-in in Firebase Authentication settings and add localhost to authorized domains. This is important for it to work.

Start the face:

npm start

This should open the app in your web browser, probably at http://localhost:3000.

How to Use It
Once both the backend (Python server) and the frontend (React app) are running:

Open your browser to http://localhost:3000.

Fill out the form with your health numbers and lifestyle info.

Click the "Get My Health Report" button.

The app will send your data to the Python brain, get the risk predictions back, and then show you your risks and some helpful recommendations.

What's Inside (Modules, simple talk)
Data Collector: Gathers and cleans up all the health info.

Risk Predictor: The smart part that uses math to guess your risk for diseases.

Trend Watcher: Keeps an eye on your health numbers over time (though this version is more about current status).

Advice Giver: Gives you tips on diet, exercise, and habits based on your risks.

Progress Tracker: Shows you how you're doing (saves your latest info).

How it's all put together: Just a way to make sure all parts talk to each other nicely.

Hope this helps you out! It was a fun project to build.
