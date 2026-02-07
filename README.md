 Local Backend Setup


# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

Activate virtual environment
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt

Configure environment
cp .env.example .env

Download/Place trained model
 Copy efficientnet_final.h5 to ./models/ directory

#Run backend server
python main.py
```

Backend will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`
--------------------------------------------------------------------------------------------------###################################################----------------------------------------------------------------
Deploy Backend to Render

1. Prepare Repository:
   # Ensure Dockerfile is in deployment/
   # Update requirements.txt
   # Set up .env.production
   

2. Create Render Service:
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repo
   - Configure:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - Set environment variables from `.env.production`

3. Deploy:
   # Render auto-deploys on git push
   git push origin main

Your backend URL will be like: `https://your-app-name.render.com`


1. Prepare Repository:
   # Ensure Dockerfile is in deployment/
   # Update requirements.txt
   # Set up .env.production
   

2. Create Render Service:
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repo
   - Configure:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - Set environment variables from `.env.production`

3. Deploy:
   # Render auto-deploys on git push
   git push origin main

Your backend URL will be like: `https://your-app-name.render.com`






