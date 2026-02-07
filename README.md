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





