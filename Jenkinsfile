pipeline {
    agent any

    environment {
        PYTHON_ENV = "venv"
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout code from Git
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh 'python3 -m venv ${PYTHON_ENV}'
                sh '${PYTHON_ENV}/bin/pip install -r requirements.txt'
            }
        }
        
        stage('Lint & Test') {
            steps {
                // Run tests using pytest or another testing framework
                echo "Running unit tests (placeholder)..."
            }
        }

        stage('Train Model') {
            steps {
                echo "Training AutoML model..."
                sh '${PYTHON_ENV}/bin/python train.py'
            }
        }
        
        stage('Package Model (BentoML)') {
            steps {
                echo "Registering model to local BentoML store..."
                sh '${PYTHON_ENV}/bin/python save_bento_model.py'
                
                echo "Building Bento container..."
                sh '${PYTHON_ENV}/bin/python -m bentoml build'
            }
        }
    }
    
    post {
        always {
            echo "Pipeline execution finished."
        }
        success {
            echo "Pipeline succeeded! Artifacts are ready for deployment."
        }
        failure {
            echo "Pipeline failed! Please check logs."
        }
    }
}
